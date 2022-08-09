#  Copyright (c) 2017-2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import division

import hashlib
import operator

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from pyarrow.parquet import ParquetFile

from petastorm.cache import NullCache
from petastorm.workers_pool import EmptyResultError
from petastorm.workers_pool.worker_base import WorkerBase


class ArrowReaderWorkerResultsQueueReader(object):
    def __init__(self):
        pass

    @property
    def batched_output(self):
        return True

    def read_next(self, workers_pool, schema, ngram):
        try:
            assert not ngram, 'ArrowReader does not support ngrams for now'

            result_table = workers_pool.get_results()

            # Convert arrow table columns into numpy. Strings are handled differently since to_pandas() returns
            # numpy array of dtype=object.
            result_dict = dict()
            for column_name in result_table.column_names:
                column = result_table.column(column_name)
                # Assume we get only one chunk since reader worker reads one rowgroup at a time

                # `to_pandas` works slower when called on the entire `data` rather directly on a chunk.
                if result_table.column(0).num_chunks == 1:
                    column_as_pandas = column.chunks[0].to_pandas()
                else:
                    column_as_pandas = column.to_pandas()

                # pyarrow < 0.15.0 would always return a numpy array. Starting 0.15 we get pandas series, hence we
                # convert it into numpy array
                if isinstance(column_as_pandas, pd.Series):
                    column_as_numpy = column_as_pandas.values
                else:
                    column_as_numpy = column_as_pandas

                if pa.types.is_string(column.type):
                    result_dict[column_name] = column_as_numpy.astype(np.unicode_)
                elif pa.types.is_list(column.type):
                    # Assuming all lists are of the same length, hence we can collate them into a matrix
                    list_of_lists = column_as_numpy
                    try:
                        col_data = np.vstack(list_of_lists.tolist())
                        shape = schema.fields[column_name].shape
                        if len(shape) > 1:
                            col_data = col_data.reshape((len(list_of_lists),) + shape)
                        result_dict[column_name] = col_data

                    except ValueError:
                        raise RuntimeError('Length of all values in column \'{}\' are expected to be the same length. '
                                           'Got the following set of lengths: \'{}\''
                                           .format(column_name,
                                                   ', '.join(str(value.shape[0]) for value in list_of_lists)))
                else:
                    result_dict[column_name] = column_as_numpy

            return schema.make_namedtuple(**result_dict)

        except EmptyResultError:
            raise StopIteration


class ArrowReaderWorker(WorkerBase):
    def __init__(self, worker_id, publish_func, args):
        super(ArrowReaderWorker, self).__init__(worker_id, publish_func, args)

        self._filesystem = args[0]
        self._dataset_path_or_paths = args[1]
        self._schema = args[2]
        self._ngram = args[3]
        self._split_pieces = args[4]
        self._local_cache = args[5]
        self._transform_spec = args[6]
        self._transformed_schema = args[7]
        self._arrow_filters = args[8]
        self._shuffle_rows = args[9]
        self._random_state = np.random.RandomState(seed=args[10])

        if self._ngram:
            raise NotImplementedError('ngrams are not supported by ArrowReaderWorker')

        # We create datasets lazily in the first invocation of 'def process'. This speeds up startup time since
        # all Worker constructors are serialized
        self._dataset = None

    @staticmethod
    def new_results_queue_reader():
        return ArrowReaderWorkerResultsQueueReader()

    # pylint: disable=arguments-differ
    def process(self, piece_index, worker_predicate, shuffle_row_drop_partition):
        """Main worker function. Loads and returns all rows matching the predicate from a rowgroup

        Looks up the requested piece (a single row-group in a parquet file). If a predicate is specified,
        columns needed by the predicate are loaded first. If no rows in the rowgroup matches the predicate criteria
        the rest of the columns are not loaded.

        :param piece_index:
        :param shuffle_row_drop_partition: A tuple 2 of the current row drop partition and the total number
            of partitions.
        :return:
        """

        if not self._dataset:
            self._dataset = pq.ParquetDataset(
                self._dataset_path_or_paths,
                filesystem=self._filesystem,
                validate_schema=False, filters=self._arrow_filters)

        piece = self._split_pieces[piece_index]

        # Create pyarrow file system
        parquet_file = ParquetFile(self._dataset.fs.open(piece.path))

        if not isinstance(self._local_cache, NullCache):
            if worker_predicate:
                raise RuntimeError('Local cache is not supported together with predicates, '
                                   'unless the dataset is partitioned by the column the predicate operates on.')
            if shuffle_row_drop_partition[1] != 1:
                raise RuntimeError('Local cache is not supported together with shuffle_row_drop_partitions > 1')

        if worker_predicate:
            all_cols = self._load_rows_with_predicate(parquet_file, piece, worker_predicate, shuffle_row_drop_partition)
        else:
            # Using hash of the dataset path with the relative path in order to:
            #  1. Make sure if a common cache serves multiple processes (e.g. redis), we don't have conflicts
            #  2. Dataset path is hashed, to make sure we don't create too long keys, which maybe incompatible with
            #     some cache implementations
            #  3. Still leave relative path and the piece_index in plain text to make it easier to debug
            if isinstance(self._dataset_path_or_paths, list):
                path_str = ','.join(self._dataset_path_or_paths)
            else:
                path_str = self._dataset_path_or_paths
            cache_key = '{}:{}:{}'.format(hashlib.md5(path_str.encode('utf-8')).hexdigest(),
                                          piece.path, piece_index)
            all_cols = self._local_cache.get(cache_key,
                                             lambda: self._load_rows(parquet_file, piece, shuffle_row_drop_partition))

        if all_cols:
            self.publish_func(all_cols)

    @staticmethod
    def _check_shape_and_ravel(x, field):
        if not isinstance(x, np.ndarray):
            raise ValueError('field {name} must be numpy array type.'.format(name=field.name))
        if None in field.shape:
            raise ValueError(f'All dimensions of a shape: {field.shape} in: {field.name} field must be constant. '
                             f'If a dimension is variable, we won\'t be able to coalesce rows when preparing a batch.')

        if x.shape != tuple(field.shape):
            raise ValueError('field {name} must be the shape {shape}'
                             .format(name=field.name, shape=field.shape))
        if not x.flags.c_contiguous:
            raise ValueError('field {name} error: only support row major multi-dimensional array.'
                             .format(name=field.name))
        return x.ravel()

    def _load_rows(self, pq_file, piece, shuffle_row_drop_range):
        """Loads all rows from a piece"""

        column_names_in_schema = set(field.name for field in self._schema.fields.values())

        result = self._read_with_shuffle_row_drop(piece, pq_file, column_names_in_schema, shuffle_row_drop_range)

        if self._transform_spec:
            result_as_pandas = result.to_pandas()
            # A user may omit `func` value if they intend just to delete some fields using the TransformSpec
            if self._transform_spec.func:
                transformed_result = self._transform_spec.func(result_as_pandas)
            else:
                transformed_result = result_as_pandas

            # If transform function left a field that is listed in transform_spec's remove_fields, we remove it
            # ourselves. Allows for the following transform-spec objects to be created:
            # TransformSpec(removed_fields=['some field'])
            for field_to_remove in set(transformed_result.columns) & set(self._transform_spec.removed_fields):
                del transformed_result[field_to_remove]

            transformed_result_column_set = set(transformed_result.columns)
            transformed_schema_column_set = set([f.name for f in self._transformed_schema.fields.values()])

            if transformed_result_column_set != transformed_schema_column_set:
                raise ValueError('Transformed result columns ({rc}) do not match required schema columns({sc})'
                                 .format(rc=','.join(transformed_result_column_set),
                                         sc=','.join(transformed_schema_column_set)))

            # For fields return multidimensional array, we need to ravel them
            # because pyarrow do not support multidimensional array.
            # later we will reshape it back.
            for field in self._transformed_schema.fields.values():
                if len(field.shape) > 1:
                    transformed_result[field.name] = transformed_result[field.name] \
                        .map(lambda x, f=field: self._check_shape_and_ravel(x, f))

            result = pa.Table.from_pandas(transformed_result, preserve_index=False)

        return result

    def _load_rows_with_predicate(self, pq_file, piece, worker_predicate, shuffle_row_drop_partition):
        """Loads all rows that match a predicate from a piece"""

        # 1. Read all columns needed by predicate
        # 2. Apply the predicate. If nothing matches, exit early
        # 3. Read the remaining columns

        # Split all column names into ones that are needed by predicateand the rest.
        predicate_column_names = set(worker_predicate.get_fields())

        if not predicate_column_names:
            raise ValueError('At least one field name must be returned by predicate\'s get_field() method')

        all_schema_names = set(field.name for field in self._schema.fields.values())

        invalid_column_names = predicate_column_names - all_schema_names
        if invalid_column_names:
            raise ValueError('At least some column names requested by the predicate ({}) '
                             'are not valid schema names: ({})'.format(', '.join(invalid_column_names),
                                                                       ', '.join(all_schema_names)))

        # Split into 'columns for predicate evaluation' and 'other columns'. We load 'other columns' only if at
        # least one row in the rowgroup matched the predicate
        other_column_names = all_schema_names - predicate_column_names

        # Read columns needed for the predicate
        predicates_table = self._read_with_shuffle_row_drop(piece, pq_file, predicate_column_names,
                                                            shuffle_row_drop_partition)

        predicates_data_frame = predicates_table.to_pandas()

        match_predicate_mask = worker_predicate.do_include(predicates_data_frame)
        erase_mask = match_predicate_mask.map(operator.not_)

        # Don't have anything left after filtering? Exit early.
        if erase_mask.all():
            return []

        predicates_data_frame[erase_mask] = None

        if other_column_names:
            # Read remaining columns
            other_table = self._read_with_shuffle_row_drop(piece, pq_file, other_column_names,
                                                           shuffle_row_drop_partition)
            other_data_frame = other_table.to_pandas()
            other_data_frame[erase_mask] = None

            # Partition-by columns will appear in both other and predicate data frames. Deduplicate.
            columns_from_predicates = predicates_data_frame.columns.difference(other_data_frame.columns)
            result_data_frame = pd.merge(predicates_data_frame[columns_from_predicates], other_data_frame,
                                         copy=False, left_index=True, right_index=True)
        else:
            result_data_frame = predicates_data_frame

        result = result_data_frame[match_predicate_mask]

        if self._transform_spec:
            result = self._transform_spec.func(result)

        return pa.Table.from_pandas(result, preserve_index=False)

    def _read_with_shuffle_row_drop(self, piece, pq_file, column_names, shuffle_row_drop_partition):
        partition_names = self._dataset.partitions.partition_names if self._dataset.partitions else set()

        # pyarrow would fail if we request a column names that the dataset is partitioned by
        table = piece.read(columns=column_names - partition_names, partitions=self._dataset.partitions)
        if self._shuffle_rows:
            indices = self._random_state.permutation(table.num_rows)
            table = table.take(indices)

        # Drop columns we did not explicitly request. This may happen when a table is partitioned. Besides columns
        # requested, pyarrow will also return partition values. Having these unexpected fields will break some
        # downstream code.

        loaded_column_names = set(table.column_names)
        unasked_for_columns = loaded_column_names - column_names
        if unasked_for_columns:
            table = table.drop(unasked_for_columns)

        num_rows = len(table)
        num_partitions = shuffle_row_drop_partition[1]
        this_partition = shuffle_row_drop_partition[0]

        if num_partitions > 1:
            data_frame_pandas = table.to_pandas()
            partition_indexes = np.floor(np.arange(num_rows) / (float(num_rows) / min(num_rows, num_partitions)))

            table = pa.Table.from_pandas(data_frame_pandas.loc[partition_indexes == this_partition],
                                         preserve_index=False)

        return table
