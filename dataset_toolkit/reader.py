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

import hashlib
import logging
import os
import warnings

from urlparse import urlparse, urlunparse
from pyarrow import parquet as pq
from pyarrow.parquet import ParquetFile

from dataset_toolkit import PredicateBase, RowGroupSelectorBase
from dataset_toolkit import utils
from dataset_toolkit.cache import NullCache
from dataset_toolkit.etl import dataset_metadata, rowgroup_indexing
from dataset_toolkit.fs_utils import FilesystemResolver
from dataset_toolkit.workers_pool import EmptyResultError
from dataset_toolkit.workers_pool.thread_pool import ThreadPool
from dataset_toolkit.workers_pool.worker_base import WorkerBase
from dataset_toolkit.workers_pool.ventilator import ConcurrentVentilator


logger = logging.getLogger(__name__)


class Reader(object):
    """Reads a unischema based dataset from a parquet file."""

    def __init__(self, schema_fields=None, dataset_url=None, shuffle=True, predicate=None, rowgroup_selector=None,
                 reader_pool=None, num_epochs=1, sequence=None, training_partition=None, num_training_partitions=None,
                 read_timeout_s=None, cache=None):
        """Initializes a reader object.

        :param schema_fields: list of unischema fields to subset, or None to read all fields.
        :param dataset_url: an filepath or a url to a parquet directory,
                       e.g. 'hdfs://some_hdfs_cluster/user/yevgeni/parquet8', or '/tmp/mydataset'
        :param predicate: instance of predicate object to filter rows to be returned by reader.
        :param rowgroup_selector: instance of row group selector object to select row groups to be read
        :param reader_pool: parallelization pool. ThreadPool(10) (10 threads) is used by default.
                       This pool is a custom implementation used to parallelize reading data from the dataset.
                       Any object from workers_pool package can be used (e.g. ProcessPool)
        :param num_epochs: An epoch is a single pass over all samples in the dataset. Setting num_epochs to 'None' will
                       result in an infinite number of epochs.
        :param sequence: If it is set to a Sequence object, then will fetch will return a sequence, otherwise fetch
                       will return an item.
        :param training_partition: An int denoting the partition number used for multi node training. Each node should
                       pass in a unique partition number in the range [0, num_training_partitions).
                       num_training_partitions must be supplied as well.
        :param num_training_partitions An int denoting the number of training partitions (how many nodes are performing
                       the multi node training)
        :param read_timeout_s: A numeric with the amount of time in seconds you would like to give a read before it
                       times out and raises an EmptyResultError. Pass in None for an infinite timeout
        :param cache: An object conforming to `cache.CacheBase` interface. Before loading row groups from a parquet file
                       the Reader will attempt to load these values from cache. Caching is useful when communication
                       to the main data store is either slow or expensive and the local machine has large enough storage
                       to store entire dataset (or a partition of a dataset if num_training_partitions is used).

        By default, `NullCache` implementation
        """
        self.sequence = sequence

        if cache is None:
            cache = NullCache()

        if not reader_pool:
            reader_pool = ThreadPool(10)

        if dataset_url and dataset_url[-1] == '/':
            dataset_url = dataset_url[:-1]

        # Create pyarrow file system
        logger.debug('dataset_url: {}'.format(dataset_url))
        resolver = FilesystemResolver(dataset_url)

        dataset = pq.ParquetDataset(resolver.parsed_dataset_url().path, filesystem=resolver.filesystem(),
                                    validate_schema=False)

        split_pieces = dataset_metadata.load_rowgroup_split(dataset)
        stored_schema = dataset_metadata.get_schema(dataset)
        if schema_fields:
            # Make a subset view, checking for invalid fields in the process
            self.schema = stored_schema.create_schema_view(schema_fields)
        else:
            self.schema = stored_schema

        if predicate:
            if not isinstance(predicate, PredicateBase):
                raise ValueError('predicate parameter is expected to be derived from PredicateBase')
            predicate_fields = predicate.get_fields()

            if set(predicate_fields) == dataset.partitions.partition_names:
                assert len(dataset.partitions.partition_names) == 1, \
                    'Datasets with only a single partition level supported at the moment'

                piece_indexes = []
                for piece_index, piece in enumerate(split_pieces):
                    partition_name, partition_index = piece.partition_keys[0]
                    partition_value = dataset.partitions[0].keys[partition_index]
                    if predicate.do_include({partition_name: partition_value}):
                        piece_indexes.append(piece_index)

                worker_predicate = None
            else:
                piece_indexes = range(len(split_pieces))
                worker_predicate = predicate

        else:
            piece_indexes = range(len(split_pieces))
            worker_predicate = None

        if rowgroup_selector:
            if not isinstance(rowgroup_selector, RowGroupSelectorBase):
                raise ValueError('rowgroup_selector parameter is expected to be derived from RowGroupSelectorBase')

            available_row_group_indexes = rowgroup_indexing.get_row_group_indexes(dataset)
            required_indexes = rowgroup_selector.get_index_names()

            if not set(required_indexes).issubset(set(available_row_group_indexes.keys())):
                raise ValueError('Some of required indexes {} are not available in {}'.format(
                    required_indexes, available_row_group_indexes.keys()))

            selected_indexes = rowgroup_selector.select_row_groups(available_row_group_indexes)
            # include only selected_indexes but in piece_indexes order
            piece_indexes = [idx for idx in piece_indexes if idx in selected_indexes]

        if training_partition is not None or num_training_partitions is not None:
            if not num_training_partitions or not isinstance(
                    training_partition, int) or not isinstance(num_training_partitions, int):
                raise ValueError('partition and num_partitions must be ints and both specified to use partitioning')
            # We hash on the relative path of each parquet file to guarantee consistency between different reader
            # constructions even after moving the dataset
            piece_indexes = filter(
                lambda index: hash(os.path.relpath(split_pieces[index].path, dataset.paths)) %
                num_training_partitions == training_partition,
                piece_indexes)

        self._workers_pool = reader_pool

        items_to_ventilate = [{'piece_index': piece_index, 'worker_predicate': worker_predicate}
                              for piece_index in piece_indexes]
        ventilator = ConcurrentVentilator(self._workers_pool.ventilate, items_to_ventilate,
                                          iterations=num_epochs, randomize_item_order=shuffle)

        # dataset_url_parsed does not go well through pickling for some reason and comes out as collections.ParseResult
        # which has no hostname/port (looses the ResultMixin). We pass string url which is safer
        self._workers_pool.start(Worker,
                                 (dataset_url, self.schema, sequence, split_pieces, cache, worker_predicate),
                                 ventilator=ventilator)
        self._read_timeout_s = read_timeout_s

    def stop(self):
        """Stops all worker threads/processes"""
        self._workers_pool.stop()

    def join(self):
        """Joins all worker threads/processes. Will block until all worker workers have been fully terminated"""
        self._workers_pool.join()

    def fetch(self, timeout=None):
        warning_message = 'fetch is deprecated. Please use iterator api to fetch data instead.'
        warnings.warn(warning_message, DeprecationWarning)
        # Since warnings are generally ignored in av, print out a logging warning as well
        logger.warn(warning_message)
        return self._workers_pool.get_results(timeout=timeout)

    def __iter__(self):
        return self

    def next(self):
        try:
            return self._workers_pool.get_results(timeout=self._read_timeout_s)
        except EmptyResultError:
            raise StopIteration

    # Functions needed to treat reader as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.join()


def _merge_two_dicts(a, b):
    """Merges two dictionaries together. If the same key is present in both input dictionaries, the value from 'y'
    dominates."""
    result = a.copy()
    result.update(b)
    return result


def _select_cols(a_dict, keys):
    """Filteres out entries in a dictionary that have a key which is not part of 'keys' argument. `a_dict` is not
    modified and a new dictionary is returned."""
    if keys == a_dict.keys():
        return a_dict
    else:
        return {field_name: a_dict[field_name] for field_name in keys}


class Worker(WorkerBase):
    def __init__(self, worker_id, publish_func, args):
        super(Worker, self).__init__(worker_id, publish_func, args)

        self._dataset_url_parsed = urlparse(args[0])
        self._schema = args[1]
        self._sequence = args[2]
        self._split_pieces = args[3]
        self._local_cache = args[4]

        # We create datasets lazily in the first invocation of 'def process'. This speeds up startup time since
        # all Worker constructors are serialized
        self._dataset = None

    def process(self, piece_index, worker_predicate):
        """Main worker function. Loads and returns all rows matching the predicate from a rowgroup

        Looks up the requested piece (a single row-group in a parquet file). If a predicate is specified,
        columns needed by the predicate are loaded first. If no rows in the rowgroup matches the predicate criteria
        the rest of the columns are not loaded.

        :param piece_index:
        :return:
        """

        if not self._dataset:
            resolver = FilesystemResolver(self._dataset_url_parsed)
            self._dataset = pq.ParquetDataset(
                resolver.parsed_dataset_url().path,
                filesystem=resolver.filesystem(),
                validate_schema=False)

        piece = self._split_pieces[piece_index]

        # Create pyarrow file system
        parquet_file = ParquetFile(self._dataset.fs.open(piece.path))

        if worker_predicate:
            if not isinstance(self._local_cache, NullCache):
                raise RuntimeError('Local cache is not supported together with predicates, '
                                   'unless the dataset is partitioned by the column the predicate operates on.')
            all_cols = self._load_rows_with_predicate(parquet_file, piece, worker_predicate)
        else:
            # Using hash of the dataset url with the relative path in order to:
            #  1. Make sure if a common cache serves multiple processes (e.g. redis), we don't have conflicts
            #  2. Dataset url is hashed, to make sure we don't create too long keys, which maybe incompatible with
            #     some cache implementations
            #  3. Still leave relative path and the piece_index in plain text to make it easier to debug
            cache_key = '{}:{}:{}'.format(hashlib.md5(urlunparse(self._dataset_url_parsed)).hexdigest(),
                                          piece.path, piece_index)
            all_cols = self._local_cache.get(cache_key, lambda: self._load_rows(parquet_file, piece))

        all_cols_as_tuples = [self._schema.make_namedtuple(**row) for row in all_cols]

        if self._sequence:
            all_cols_as_tuples = self._sequence.form_sequence(data=all_cols_as_tuples)

        for item in all_cols_as_tuples:
            self.publish_func(item)

    def _load_rows(self, file, piece):
        """Loads all rows from a piece"""

        # pyarrow would fail if we request a column names that the dataset is partitioned by, so we strip them from
        # the `columns` argument.
        partitions = self._dataset.partitions
        column_names = set(field.name for field in self._schema.fields.values()) - partitions.partition_names

        all_rows = piece.read(open_file_func=lambda _: file, columns=column_names, partitions=partitions) \
            .to_pandas() \
            .to_dict('records')

        return [utils.decode_row(row, self._schema) for row in all_rows]

    def _load_rows_with_predicate(self, file, piece, worker_predicate):
        """Loads all rows that match a predicate from a piece"""

        # 1. Read all columns needed by predicate and decode
        # 2. Apply the predicate. If nothing matches, exit early
        # 3. Read the remaining columns and decode
        # 4. Combine with columns already decoded for the predicate.

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

        other_column_names = all_schema_names - predicate_column_names - \
            self._dataset.partitions.partition_names

        # Read columns needed for the predicate
        predicate_rows = piece.read(
            open_file_func=lambda _: file,
            columns=predicate_column_names,
            partitions=self._dataset.partitions).to_pandas().to_dict('records')

        # Decode values
        decoded_predicate_rows = [utils.decode_row(_select_cols(row, predicate_column_names), self._schema)
                                  for row in predicate_rows]

        # Use the predicate to filter
        match_predicate_mask = [worker_predicate.do_include(row) for row in decoded_predicate_rows]

        # Don't have anything left after filtering? Exit early.
        if not any(match_predicate_mask):
            return []

        # Remove rows that were filtered out by the predicate
        filtered_decoded_predicate_rows = [row for i, row in enumerate(decoded_predicate_rows) if
                                           match_predicate_mask[i]]

        if other_column_names:
            # Read remaining columns
            other_rows = piece.read(
                open_file_func=lambda filepath: file,
                columns=other_column_names,
                partitions=self._dataset.partitions).to_pandas().to_dict('records')

            # Remove rows that were filtered out by the predicate
            filtered_other_rows = [row for i, row in enumerate(other_rows) if match_predicate_mask[i]]

            # Decode remaining columns
            decoded_other_rows = [utils.decode_row(row, self._schema) for row in filtered_other_rows]

            # Merge predicate needed columns with the remaining
            all_cols = [_merge_two_dicts(a, b) for a, b in zip(decoded_other_rows, filtered_decoded_predicate_rows)]
            return all_cols
        else:
            return filtered_decoded_predicate_rows
