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

import collections
import logging
import os
import warnings

from pyarrow import parquet as pq

from petastorm import PredicateBase, RowGroupSelectorBase
from petastorm.cache import NullCache
from petastorm.etl import dataset_metadata, rowgroup_indexing
from petastorm.fs_utils import FilesystemResolver
from petastorm.ngram import NGram
from petastorm.reader_worker import ReaderWorker
from petastorm.workers_pool import EmptyResultError
from petastorm.workers_pool.thread_pool import ThreadPool
from petastorm.workers_pool.ventilator import ConcurrentVentilator

logger = logging.getLogger(__name__)


class ShuffleOptions(object):
    shuffle_row_groups = None
    shuffle_row_drop_partitions = None

    def __init__(self, shuffle_row_groups=True, shuffle_row_drop_partitions=1):
        """
        Constructor.

        :param shuffle_row_groups: Whether to shuffle row groups (the order in which full row groups are read)
        :param shuffle_row_drop_partitions: This is is a positive integer which determines how many partitions to
            break up a row group into for increased shuffling in exchange for worse performance (extra reads).
            For example if you specify 2 each row group read will drop half of the rows within every row group and
            read the remaining rows in separate reads. It is recommended to keep this number below the regular row
            group size in order to not waste reads which drop all rows.
        """
        if not isinstance(shuffle_row_drop_partitions, int) or not shuffle_row_drop_partitions >= 1:
            raise ValueError('shuffle_row_drop_ratio must be positive integer')
        self.shuffle_row_groups = shuffle_row_groups
        self.shuffle_row_drop_partitions = shuffle_row_drop_partitions


class Reader(object):
    """Reads a unischema based dataset from a parquet file."""

    def __init__(self, dataset_url, schema_fields=None, shuffle=None, predicate=None, rowgroup_selector=None,
                 reader_pool=None, num_epochs=1, sequence=None, training_partition=None, num_training_partitions=None,
                 read_timeout_s=None, cache=None, shuffle_options=None):
        """Initializes a reader object.

        :param dataset_url: an filepath or a url to a parquet directory,
                       e.g. 'hdfs://some_hdfs_cluster/user/yevgeni/parquet8', or '/tmp/mydataset'
        :param schema_fields:
            Either list of unischema fields to subset, or None to read all fields.
            OR an NGram object, then it will return an NGram of the specified properties.
        :param predicate: instance of predicate object to filter rows to be returned by reader.
        :param rowgroup_selector: instance of row group selector object to select row groups to be read
        :param reader_pool: parallelization pool. ThreadPool(10) (10 threads) is used by default.
                       This pool is a custom implementation used to parallelize reading data from the dataset.
                       Any object from workers_pool package can be used (e.g. ProcessPool)
        :param num_epochs: An epoch is a single pass over all samples in the dataset. Setting num_epochs to 'None' will
                       result in an infinite number of epochs.
        :param sequence: This is deprecated. To use sequence/ngram, please supply the argument in schema_fields instead.
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
        :param shuffle_options : ShuffleOptions object to describe how to shuffle dataset (supercedes shuffle parameter)
                       defaults to shuffling row groups but not to drop rows based on partitions.
        :param shuffle: DEPRECATED boolean whether to shuffle the row group order. Use shuffle_row_groups in
                       ShuffleOptions instead.

        By default, `NullCache` implementation
        """

        # 1. Resolve dataset path (hdfs://, file://) and open the parquet storage (dataset)
        # 2. Get a list of all groups
        # 3. Filter rowgroups
        #    a. predicates
        #    b. row-group selector (our indexing mechanism)
        #    c. partition: used to get a subset of data for distributed training
        # 4. Create a rowgroup ventilator object
        # 5. Start workers pool
        if not (isinstance(schema_fields, collections.Iterable) or isinstance(schema_fields, NGram)
                or schema_fields is None):
            raise ValueError("""Fields must be either None, an iterable collection of Unischema fields or an NGram
            object.""")

        if sequence is not None:
            raise ValueError("""'sequence' argument of Reader object is deprecated. Please pass an NGram instance to
            'schema_fields' argument instead.""")

        self.ngram = schema_fields if isinstance(schema_fields, NGram) else None

        cache = cache or NullCache()
        dataset_url = dataset_url[:-1] if dataset_url[-1] == '/' else dataset_url
        self._workers_pool = reader_pool or ThreadPool(10)

        # 1. Resolve dataset path (hdfs://, file://) and open the parquet storage (dataset)
        logger.debug('dataset_url: %s', dataset_url)
        resolver = FilesystemResolver(dataset_url)
        dataset = pq.ParquetDataset(resolver.parsed_dataset_url().path, filesystem=resolver.filesystem(),
                                    validate_schema=False)

        # Get a unischema stored in the dataset metadata.
        stored_schema = dataset_metadata.get_schema(dataset)

        # Make a schema view (a view is a Unischema containing only a subset of fields
        # Will raise an exception if invalid schema fields are in schema_fields
        fields = schema_fields if isinstance(schema_fields, collections.Iterable) else None
        self.schema = stored_schema.create_schema_view(fields) if fields else stored_schema

        # 2. Get a list of all groups
        row_groups = dataset_metadata.load_row_groups(dataset)

        # 3. Filter rowgroups
        filtered_row_group_indexes, worker_predicate = self._filter_row_groups(dataset, row_groups, predicate,
                                                                               rowgroup_selector, training_partition,
                                                                               num_training_partitions)
        # 4. Create a rowgroup ventilator object
        if shuffle_options is None:
            if shuffle is None:
                shuffle = True
            else:
                logger.warning('shuffle option is deprecated. Please use shuffle_options instead')
            shuffle_options = ShuffleOptions(shuffle)
        ventilator = self._create_ventilator(filtered_row_group_indexes, shuffle_options, num_epochs, worker_predicate)

        # 5. Start workers pool
        self._workers_pool.start(ReaderWorker,
                                 (dataset_url, self.schema, self.ngram, row_groups, cache, worker_predicate),
                                 ventilator=ventilator)
        self._read_timeout_s = read_timeout_s

    def _filter_row_groups(self, dataset, row_groups, predicate, rowgroup_selector, training_partition,
                           num_training_partitions):
        """Calculates which rowgroups will be read during.

        The following filters are applied: predicates;  row-group selector (our indexing mechanism); training partition

        :param dataset: ParquetDataset instance
        :param row_groups: a list of row groups (a list of ParquetDatasetPiece objects)
        :param predicate: instance of predicate object to filter rows to be returned by reader.
        :param rowgroup_selector: instance of row group selector object to select row groups to be read
        :param training_partition: An int denoting the partition number used for multi node training. Each node should
                       pass in a unique partition number in the range [0, num_training_partitions).
                       num_training_partitions must be supplied as well.
        :param num_training_partitions An int denoting the number of training partitions (how many nodes are performing
                       the multi node training)
        :return: (filtered_row_group_indexes, worker_predicate): filtered_row_group_indexes an integer index into
        row_groups array. worker_predicate contains only predicates that could not be resolved on the partitioned fields
        and need to be evaluated by workers.
        """

        filtered_row_group_indexes, worker_predicate = \
            self._apply_predicate_to_row_groups(dataset, row_groups, predicate)

        if rowgroup_selector:
            filtered_row_group_indexes = self._apply_row_group_selector(dataset, rowgroup_selector,
                                                                        filtered_row_group_indexes)

        if training_partition is not None or num_training_partitions is not None:
            filtered_row_group_indexes = self._partition_row_groups(dataset, row_groups, num_training_partitions,
                                                                    training_partition,
                                                                    filtered_row_group_indexes)
        return filtered_row_group_indexes, worker_predicate

    def _partition_row_groups(self, dataset, row_groups, num_training_partitions, training_partition,
                              filtered_row_group_indexes):
        """Filters the list of row group indexes based on the requested training partitions. Returns
        a modified list of rowgroup indexes."""

        if not num_training_partitions \
                or not isinstance(training_partition, int) \
                or not isinstance(num_training_partitions, int):
            raise ValueError('partition and num_partitions must be ints and both specified to use partitioning')

        # We hash on the relative path of each parquet file to guarantee consistency between different reader
        # constructions even after moving the dataset
        filtered_row_group_indexes = [index for index in filtered_row_group_indexes
                                      if hash(os.path.relpath(row_groups[index].path, dataset.paths)) %
                                      num_training_partitions == training_partition]
        return filtered_row_group_indexes

    def _apply_row_group_selector(self, dataset, rowgroup_selector, filtered_row_group_indexes):
        """Filters the list of row group indexes using rowgroup selector object. Returns a modified list of rowgroup
        indexes."""

        if not isinstance(rowgroup_selector, RowGroupSelectorBase):
            raise ValueError('rowgroup_selector parameter is expected to be derived from RowGroupSelectorBase')

        # Load indexes from metadata
        available_row_group_indexes = rowgroup_indexing.get_row_group_indexes(dataset)

        required_indexes = rowgroup_selector.get_index_names()
        if not set(required_indexes).issubset(set(available_row_group_indexes.keys())):
            raise ValueError('Some of required indexes {} are not available in {}'.format(
                required_indexes, list(available_row_group_indexes.keys())))

        selected_indexes = rowgroup_selector.select_row_groups(available_row_group_indexes)

        # include only selected_indexes but in filtered_row_group_indexes order
        filtered_row_group_indexes = [idx for idx in filtered_row_group_indexes if idx in selected_indexes]
        return filtered_row_group_indexes

    def _apply_predicate_to_row_groups(self, dataset, row_groups, predicate):
        """Filters the list of row group indexes using rowgroup selector object. Returns a modified list of rowgroup
        indexes and a list of worker_predicate: predicates that could not be applied at this level
        (parquet partitioning)"""

        if predicate:
            if not isinstance(predicate, PredicateBase):
                raise ValueError('predicate parameter is expected to be derived from PredicateBase')
            predicate_fields = predicate.get_fields()

            if set(predicate_fields) == dataset.partitions.partition_names:
                assert len(dataset.partitions.partition_names) == 1, \
                    'Datasets with only a single partition level supported at the moment'

                filtered_row_group_indexes = []
                for piece_index, piece in enumerate(row_groups):
                    partition_name, partition_index = piece.partition_keys[0]
                    partition_value = dataset.partitions[0].keys[partition_index]
                    if predicate.do_include({partition_name: partition_value}):
                        filtered_row_group_indexes.append(piece_index)

                worker_predicate = None
            else:
                filtered_row_group_indexes = list(range(len(row_groups)))
                worker_predicate = predicate

        else:
            filtered_row_group_indexes = list(range(len(row_groups)))
            worker_predicate = None
        return filtered_row_group_indexes, worker_predicate

    def _create_ventilator(self, row_group_indexes, shuffle_options, num_epochs, worker_predicate):
        items_to_ventilate = []
        for piece_index in row_group_indexes:
            for shuffle_row_drop_partition in range(shuffle_options.shuffle_row_drop_partitions):
                items_to_ventilate.append(
                    {'piece_index': piece_index,
                     'worker_predicate': worker_predicate,
                     'shuffle_row_drop_partition': (shuffle_row_drop_partition,
                                                    shuffle_options.shuffle_row_drop_partitions)})

        return ConcurrentVentilator(self._workers_pool.ventilate,
                                    items_to_ventilate,
                                    iterations=num_epochs,
                                    randomize_item_order=shuffle_options.shuffle_row_groups)

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

    def __next__(self):
        try:
            return self._workers_pool.get_results(timeout=self._read_timeout_s)
        except EmptyResultError:
            raise StopIteration

    def next(self):
        return self.__next__()

    # Functions needed to treat reader as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.join()
