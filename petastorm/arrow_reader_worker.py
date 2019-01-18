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
            assert 1 == result_table.column(0).data.num_chunks

            # Convert arrow table columns into numpy. Strings are handled differently since to_pandas() returns
            # numpy array of dtype=object.
            result_dict = dict()
            for column in result_table.columns:
                # Assume we get only one chunk since reader worker reads one rowgroup at a time
                assert len(column.data.chunks) == 1
                if pa.types.is_string(column.type):
                    result_dict[column.name] = column.data.chunks[0].to_pandas().astype(np.unicode_)
                elif pa.types.is_list(column.type):
                    # Assuming all lists are of the same length, hence we can collate them into a matrix
                    list_of_lists = column.data.chunks[0].to_pandas()
                    try:
                        result_dict[column.name] = np.vstack(list_of_lists.tolist())
                    except ValueError:
                        raise RuntimeError('Length of all values in column \'%s\' are expected to be the same length. '
                                           'Got the following set of lengths: \'%s\'',
                                           column.name,
                                           ', '.join({value.shape[0] for value in list_of_lists}))
                else:
                    result_dict[column.name] = column.data.chunks[0].to_pandas()

            return schema.make_namedtuple(**result_dict)

        except EmptyResultError:
            raise StopIteration


class ArrowReaderWorker(WorkerBase):
    def __init__(self, worker_id, publish_func, args):
        super(ArrowReaderWorker, self).__init__(worker_id, publish_func, args)

        self._filesystem = args[0]
        self._dataset_path = args[1]
        self._schema = args[2]
        self._ngram = args[3]
        self._split_pieces = args[4]
        self._local_cache = args[5]
        self._transform_spec = args[6]

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
                self._dataset_path,
                filesystem=self._filesystem,
                validate_schema=False)

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
            cache_key = '{}:{}:{}'.format(hashlib.md5(self._dataset_path.encode('utf-8')).hexdigest(),
                                          piece.path, piece_index)
            all_cols = self._local_cache.get(cache_key,
                                             lambda: self._load_rows(parquet_file, piece, shuffle_row_drop_partition))

        if all_cols:
            self.publish_func(all_cols)

    def _load_rows(self, pq_file, piece, shuffle_row_drop_range):
        """Loads all rows from a piece"""

        # pyarrow would fail if we request a column names that the dataset is partitioned by, so we strip them from
        # the `columns` argument.
        partitions = self._dataset.partitions
        column_names_in_schema = set(field.name for field in self._schema.fields.values())
        column_names = column_names_in_schema - partitions.partition_names

        result = self._read_with_shuffle_row_drop(piece, pq_file, column_names, shuffle_row_drop_range)

        if self._transform_spec:
            result = pa.Table.from_pandas(self._transform_spec.func(result.to_pandas()), preserve_index=False)

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
        table = piece.read(
            open_file_func=lambda _: pq_file,
            columns=column_names,
            partitions=self._dataset.partitions
        )

        num_rows = len(table)
        num_partitions = shuffle_row_drop_partition[1]
        this_partition = shuffle_row_drop_partition[0]

        if num_partitions > 1:
            data_frame_pandas = table.to_pandas()
            partition_indexes = np.floor(np.arange(num_rows) / (float(num_rows) / min(num_rows, num_partitions)))

            table = pa.Table.from_pandas(data_frame_pandas.loc[partition_indexes == this_partition],
                                         preserve_index=False)

        return table
