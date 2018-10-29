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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import six
from pyarrow import parquet as pq

from petastorm.cache import NullCache
from petastorm.etl import dataset_metadata, rowgroup_indexing
from petastorm.fs_utils import FilesystemResolver
from petastorm.local_disk_cache import LocalDiskCache
from petastorm.ngram import NGram
from petastorm.predicates import PredicateBase
from petastorm.reader_impl.reader_v2 import ReaderV2
from petastorm.reader_impl.same_thread_executor import SameThreadExecutor
from petastorm.reader_impl.shuffling_buffer import NoopShufflingBuffer, RandomShufflingBuffer
from petastorm.reader_worker import ReaderWorker
from petastorm.selectors import RowGroupSelectorBase
from petastorm.unischema import Unischema
from petastorm.workers_pool import EmptyResultError
from petastorm.workers_pool.dummy_pool import DummyPool
from petastorm.workers_pool.process_pool import ProcessPool
from petastorm.workers_pool.thread_pool import ThreadPool
from petastorm.workers_pool.ventilator import ConcurrentVentilator

logger = logging.getLogger(__name__)

# Make it easier to import as "from petastorm.reader import ReaderV2"
ReaderV2 = ReaderV2

# Ventilator guarantees that no more than workers + _VENTILATE_EXTRA_ROWGROUPS are processed at a moment by a
# worker pool. This guarantees that we don't run out of memory if data consumer is slower than the Reader.
_VENTILATE_EXTRA_ROWGROUPS = 2


def make_reader(dataset_url,
                schema_fields=None,
                reader_pool_type='thread', workers_count=10, pyarrow_serialize=False,
                shuffle_row_groups=True, shuffle_row_drop_partitions=1,
                predicate=None,
                rowgroup_selector=None,
                num_epochs=1,
                cur_shard=None, shard_count=None,
                cache_type='null', cache_location=None, cache_size_limit=None,
                cache_row_size_estimate=None, cache_extra_settings=None,
                hdfs_driver='libhdfs3',
                infer_schema=False,
                reader_engine='reader_v1', reader_engine_params=None):
    """
    Factory convenience method for :class:`Reader`.

    :param dataset_url: an filepath or a url to a parquet directory,
        e.g. ``'hdfs://some_hdfs_cluster/user/yevgeni/parquet8'``, or ``'file:///tmp/mydataset'``
        or ``'s3://bucket/mydataset'``.
    :param schema_fields: Either list of unischema fields to subset, or ``None`` to read all fields.
            OR an NGram object, then it will return an NGram of the specified properties.
    :param reader_pool_type: A string denoting the reader pool type. Should be one of ['thread', 'process', 'dummy']
        denoting a thread pool, process pool, or running everything in the master thread. Defaults to 'thread'
    :param workers_count: An int for the number of workers to use in the reader pool. This only is used for the
        thread or process pool. Defaults to 10
    :param pyarrow_serialize: Whether to use pyarrow for serialization. Currently only applicable to process pool.
        Defaults to False.
    :param shuffle_row_groups: Whether to shuffle row groups (the order in which full row groups are read)
    :param shuffle_row_drop_partitions: This is is a positive integer which determines how many partitions to
        break up a row group into for increased shuffling in exchange for worse performance (extra reads).
        For example if you specify 2 each row group read will drop half of the rows within every row group and
        read the remaining rows in separate reads. It is recommended to keep this number below the regular row
        group size in order to not waste reads which drop all rows.
    :param predicate: instance of :class:`.PredicateBase` object to filter rows to be returned by reader.
    :param rowgroup_selector: instance of row group selector object to select row groups to be read
    :param num_epochs: An epoch is a single pass over all rows in the dataset. Setting ``num_epochs`` to
        ``None`` will result in an infinite number of epochs.
    :param cur_shard: An int denoting the current shard number. Each node reading a shard should
        pass in a unique shard number in the range [0, shard_count). shard_count must be supplied as well.
        Defaults to None
    :param shard_count: An int denoting the number of shards to break this dataset into. Defaults to None
    :param cache_type: A string denoting the cache type, if desired. Options are [None, 'null', 'local-disk'] to
        either have a null/noop cache or a cache implemented using diskcache. Caching is useful when communication
        to the main data store is either slow or expensive and the local machine has large enough storage
        to store entire dataset (or a partition of a dataset if shard_count is used). By default will be a null cache.
    :param cache_location: A string denoting the location or path of the cache.
    :param cache_size_limit: An int specifying the size limit of the cache in bytes
    :param cache_row_size_estimate: An int specifying the estimated size of a row in the dataset
    :param cache_extra_settings: A dictionary of extra settings to pass to the cache implementation,
    :param hdfs_driver: A string denoting the hdfs driver to use (if using a dataset on hdfs). Current choices are
        libhdfs (java through JNI) or libhdfs3 (C++)
    :param infer_schema: Whether to infer the unischema object from the parquet schema.
            Only works for schemas containing certain scalar type.
    :param reader_engine: Multiple engine implementations exist ('reader_v1' and 'experimental_reader_v2'). 'reader_v1'
        (the default value) selects a stable reader implementation.
    :param reader_engine_params: For advanced usage: a dictionary with arguments passed directly to a reader
        implementation constructor chosen by ``reader_engine`` argument.  You should not use this parameter, unless you
        fine-tuning of a reader.
    :return: A :class:`Reader` object
    """

    if dataset_url is None or not isinstance(dataset_url, six.string_types):
        raise ValueError("""dataset_url must be a string""")

    dataset_url = dataset_url[:-1] if dataset_url[-1] == '/' else dataset_url
    logger.debug('dataset_url: %s', dataset_url)

    resolver = FilesystemResolver(dataset_url, hdfs_driver=hdfs_driver)
    filesystem = resolver.filesystem()
    dataset_path = resolver.get_dataset_path()

    if cache_type is None or cache_type == 'null':
        cache = NullCache()
    elif cache_type == 'local-disk':
        cache = LocalDiskCache(cache_location, cache_size_limit, cache_row_size_estimate, **cache_extra_settings or {})
    else:
        raise ValueError('Unknown cache_type: {}'.format(cache_type))

    if reader_engine == 'reader_v1':
        if reader_pool_type == 'thread':
            reader_pool = ThreadPool(workers_count)
        elif reader_pool_type == 'process':
            reader_pool = ProcessPool(workers_count, pyarrow_serialize=pyarrow_serialize)
        elif reader_pool_type == 'dummy':
            reader_pool = DummyPool()
        else:
            raise ValueError('Unknown reader_pool_type: {}'.format(reader_pool_type))

        # Create a dictionary with all ReaderV2 parameters, so we can merge with reader_engine_params if specified
        kwargs = {
            'schema_fields': schema_fields,
            'reader_pool': reader_pool,
            'shuffle_row_groups': shuffle_row_groups,
            'shuffle_row_drop_partitions': shuffle_row_drop_partitions,
            'predicate': predicate,
            'rowgroup_selector': rowgroup_selector,
            'num_epochs': num_epochs,
            'cur_shard': cur_shard,
            'shard_count': shard_count,
            'cache': cache,
            'infer_schema': infer_schema,
        }

        if reader_engine_params:
            kwargs.update(reader_engine_params)

        return Reader(filesystem, dataset_path, **kwargs)
    elif reader_engine == 'experimental_reader_v2':
        if reader_pool_type == 'thread':
            decoder_pool = ThreadPoolExecutor(workers_count)
        elif reader_pool_type == 'process':
            decoder_pool = ProcessPoolExecutor(workers_count)
        elif reader_pool_type == 'dummy':
            decoder_pool = SameThreadExecutor()
        else:
            raise ValueError('Unknown reader_pool_type: {}'.format(reader_pool_type))

        # TODO(yevgeni): once ReaderV2 is ready to be out of experimental status, we should extend
        # the make_reader interfaces to take shuffling buffer parameters explicitly
        shuffling_queue = RandomShufflingBuffer(1000, 800) if shuffle_row_groups else NoopShufflingBuffer()

        # Create a dictionary with all ReaderV2 parameters, so we can merge with reader_engine_params if specified
        kwargs = {
            'schema_fields': schema_fields,
            'predicate': predicate,
            'rowgroup_selector': rowgroup_selector,
            'num_epochs': num_epochs,
            'cur_shard': cur_shard,
            'shard_count': shard_count,
            'cache': cache,
            'decoder_pool': decoder_pool,
            'shuffling_queue': shuffling_queue,
            'shuffle_row_groups': shuffle_row_groups,
            'shuffle_row_drop_partitions': shuffle_row_drop_partitions
        }

        if reader_engine_params:
            kwargs.update(reader_engine_params)

        return ReaderV2(dataset_url, **kwargs)

    else:
        raise ValueError('Unexpected value of reader_engine argument \'%s\'. '
                         'Supported reader_engine values are \'reader_v1\' and \'experimental_reader_v2\'',
                         reader_engine)


class Reader(object):
    """Reads a dataset from a Petastorm dataset.

    :ivar last_row_consumed: True if the last row was already returned by the Reader.
    """

    def __init__(self, pyarrow_filesystem, dataset_path, schema_fields=None,
                 shuffle_row_groups=True, shuffle_row_drop_partitions=1,
                 predicate=None, rowgroup_selector=None, reader_pool=None, num_epochs=1,
                 cur_shard=None, shard_count=None, cache=None, infer_schema=False):
        """Initializes a reader object.

        :param pyarrow_filesystem: An instance of ``pyarrow.FileSystem`` that will be used. If not specified,
            then a default one will be selected based on the url (only for ``hdfs://`` or ``file://``; for
            ``s3://`` support, use ``make_reader``). The default hdfs driver is ``libhdfs3``. If you want
            to to use ``libhdfs``, use
            ``pyarrow_filesystem=pyarrow.hdfs.connect('hdfs:///some/path', driver='libhdfs')``.
        :param dataset_path: filepath to a parquet directory on the specified filesystem.
            e.g. ``'/user/yevgeni/parquet8'``, or ``'/tmp/mydataset'``.
        :param schema_fields: Either list of unischema fields to subset, or ``None`` to read all fields.
            OR an NGram object, then it will return an NGram of the specified properties.
        :param shuffle_row_groups: Whether to shuffle row groups (the order in which full row groups are read)
        :param shuffle_row_drop_partitions: This is is a positive integer which determines how many partitions to
            break up a row group into for increased shuffling in exchange for worse performance (extra reads).
            For example if you specify 2 each row group read will drop half of the rows within every row group and
            read the remaining rows in separate reads. It is recommended to keep this number below the regular row
            group size in order to not waste reads which drop all rows.
        :param predicate: instance of predicate object to filter rows to be returned by reader.
        :param rowgroup_selector: instance of row group selector object to select row groups to be read
        :param reader_pool: parallelization pool. ``ThreadPool(10)`` (10 threads) is used by default.
            This pool is a custom implementation used to parallelize reading data from the dataset.
            Any object from workers_pool package can be used
            (e.g. :class:`petastorm.workers_pool.process_pool.ProcessPool`).
        :param num_epochs: An epoch is a single pass over all rows in the dataset. Setting ``num_epochs`` to
            ``None`` will result in an infinite number of epochs.
        :param cur_shard: An int denoting the current shard number used. Each reader instance should
            pass in a unique shard number in the range ``[0, shard_count)``.
            ``shard_count`` must be supplied as well. Defaults to None
        :param shard_count: An int denoting the number of shard partitions there are. Defaults to None
        :param cache: An object conforming to :class:`.CacheBase` interface. Before loading row groups from a parquet
            file the Reader will attempt to load these values from cache. Caching is useful when communication
            to the main data store is either slow or expensive and the local machine has large enough storage
            to store entire dataset (or a partition of a dataset if shards are used).
            By default, use the :class:`.NullCache` implementation.
        """

        # 1. Open the parquet storage (dataset)
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

        self.ngram = schema_fields if isinstance(schema_fields, NGram) else None

        if self.ngram and not self.ngram.timestamp_overlap and shuffle_row_drop_partitions > 1:
            raise NotImplementedError('Using timestamp_overlap=False is not implemented with'
                                      ' shuffle_options.shuffle_row_drop_partitions > 1')

        cache = cache or NullCache()

        self._workers_pool = reader_pool or ThreadPool(10)
        # 1. Resolve dataset path (hdfs://, file://) and open the parquet storage (dataset)
        self.dataset = pq.ParquetDataset(dataset_path, filesystem=pyarrow_filesystem,
                                         validate_schema=False)

        if infer_schema:
            # If inferring schema, just retrieve the schema from a file of the dataset
            meta = self.dataset.pieces[0].get_metadata(self.dataset.fs.open)
            arrow_schema = meta.schema.to_arrow_schema()
            stored_schema = Unischema.from_arrow_schema(arrow_schema)
        else:
            # Otherwise, get the stored schema
            stored_schema = dataset_metadata.get_schema(self.dataset)

        # Make a schema view (a view is a Unischema containing only a subset of fields
        # Will raise an exception if invalid schema fields are in schema_fields
        fields = schema_fields if isinstance(schema_fields, collections.Iterable) else None
        self.schema = stored_schema.create_schema_view(fields) if fields else stored_schema

        # 2. Get a list of all groups
        row_groups = dataset_metadata.load_row_groups(self.dataset)

        # 3. Filter rowgroups
        filtered_row_group_indexes, worker_predicate = self._filter_row_groups(self.dataset, row_groups, predicate,
                                                                               rowgroup_selector, cur_shard,
                                                                               shard_count)
        # 4. Create a rowgroup ventilator object
        normalized_shuffle_row_drop_partitions = \
            self._normalize_shuffle_options(shuffle_row_drop_partitions, self.dataset)
        ventilator = self._create_ventilator(filtered_row_group_indexes, shuffle_row_groups,
                                             normalized_shuffle_row_drop_partitions, num_epochs, worker_predicate,
                                             self._workers_pool.workers_count + _VENTILATE_EXTRA_ROWGROUPS)

        # 5. Start workers pool
        self._workers_pool.start(ReaderWorker,
                                 (pyarrow_filesystem, dataset_path, self.schema, self.ngram, row_groups, cache),
                                 ventilator=ventilator)
        logger.debug('Workers pool started')

        self.last_row_consumed = False

        # _result
        self._result_buffer = []

    def _filter_row_groups(self, dataset, row_groups, predicate, rowgroup_selector, cur_shard,
                           shard_count):
        """Calculates which rowgroups will be read during.

        The following filters are applied:
        - predicates;
        - row-group selector (our indexing mechanism);
        - training partition

        :param dataset: ParquetDataset instance
        :param row_groups: a list of row groups (a list of ParquetDatasetPiece objects)
        :param predicate: instance of predicate object to filter rows to be returned by reader.
        :param rowgroup_selector: instance of row group selector object to select row groups to be read
        :param cur_shard: An int denoting the current shard number used. Each node should
                       pass in a unique partition number in the range [0, shard_count).
        :param shard_count An int denoting the number of reader shards
        :return: (filtered_row_group_indexes, worker_predicate): filtered_row_group_indexes an integer index into
        row_groups array. worker_predicate contains only predicates that could not be resolved on the partitioned fields
        and need to be evaluated by workers.
        """

        filtered_row_group_indexes, worker_predicate = \
            self._apply_predicate_to_row_groups(dataset, row_groups, predicate)

        if rowgroup_selector:
            filtered_row_group_indexes = self._apply_row_group_selector(dataset, rowgroup_selector,
                                                                        filtered_row_group_indexes)

        if cur_shard is not None or shard_count is not None:
            filtered_row_group_indexes = self._partition_row_groups(dataset, row_groups, shard_count,
                                                                    cur_shard,
                                                                    filtered_row_group_indexes)
        return filtered_row_group_indexes, worker_predicate

    def _partition_row_groups(self, dataset, row_groups, shard_count, cur_shard,
                              filtered_row_group_indexes):
        """Filters the list of row group indexes based on the requested training partitions. Returns
        a modified list of rowgroup indexes."""

        if not shard_count \
                or not isinstance(cur_shard, int) \
                or not isinstance(shard_count, int):
            raise ValueError('partition and num_partitions must be ints and both specified to use partitioning')

        # We hash on the relative path of each parquet file to guarantee consistency between different reader
        # constructions even after moving the dataset
        filtered_row_group_indexes = [index for index in filtered_row_group_indexes
                                      if hash(os.path.relpath(row_groups[index].path, dataset.paths)) %
                                      shard_count == cur_shard]
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
        (parquet partitioning)."""

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

    @staticmethod
    def _normalize_shuffle_options(shuffle_row_drop_partitions, dataset):
        """Checks that shuffle_options doesnt ask for more patitions than rows in a row group.
        This prevents sending partitions to workers which will result in not reading anything."""
        if shuffle_row_drop_partitions > 1 and dataset.metadata and dataset.metadata.num_row_groups:
            max_rows_in_row_group = 1
            for i in six.moves.xrange(dataset.metadata.num_row_groups):
                max_rows_in_row_group = max(max_rows_in_row_group, dataset.metadata.row_group(i).num_rows)

            return min(shuffle_row_drop_partitions, max_rows_in_row_group)
        return shuffle_row_drop_partitions

    def _create_ventilator(self, row_group_indexes, shuffle_row_groups, shuffle_row_drop_partitions,
                           num_epochs, worker_predicate, max_ventilation_queue_size):
        items_to_ventilate = []
        for piece_index in row_group_indexes:
            for shuffle_row_drop_partition in range(shuffle_row_drop_partitions):
                items_to_ventilate.append(
                    {'piece_index': piece_index,
                     'worker_predicate': worker_predicate,
                     'shuffle_row_drop_partition': (shuffle_row_drop_partition,
                                                    shuffle_row_drop_partitions)})

        return ConcurrentVentilator(self._workers_pool.ventilate,
                                    items_to_ventilate,
                                    iterations=num_epochs,
                                    max_ventilation_queue_size=max_ventilation_queue_size,
                                    randomize_item_order=shuffle_row_groups)

    def stop(self):
        """Stops all worker threads/processes."""
        self._workers_pool.stop()

    def join(self):
        """Joins all worker threads/processes. Will block until all worker workers have been fully terminated."""
        self._workers_pool.join()

    def fetch(self, timeout=None):
        warning_message = 'fetch is deprecated. Please use iterator api to fetch data instead.'
        warnings.warn(warning_message, DeprecationWarning)
        # Since warnings are generally ignored in av, print out a logging warning as well
        logger.warn(warning_message)
        return self._workers_pool.get_results()

    @property
    def diagnostics(self):
        return self._workers_pool.diagnostics

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # We are receiving decoded rows from the worker in chunks. We store the list internally
            # and return a single item upon each consequent call to __next__
            if not self._result_buffer:
                # Reverse order, so we can pop from the end of the list in O(1) while maintaining
                # order the items are returned from the worker
                rows_as_dict = list(reversed(self._workers_pool.get_results()))

                if self.ngram:
                    for ngram_row in rows_as_dict:
                        for timestamp in ngram_row.keys():
                            row = ngram_row[timestamp]
                            schema_at_timestamp = self.ngram.get_schema_at_timestep(self.schema, timestamp)

                            ngram_row[timestamp] = schema_at_timestamp.make_namedtuple(**row)
                    self._result_buffer = rows_as_dict
                else:
                    self._result_buffer = [self.schema.make_namedtuple(**row) for row in rows_as_dict]

            return self._result_buffer.pop()

        except EmptyResultError:
            self.last_row_consumed = True
            raise StopIteration

    def next(self):
        return self.__next__()

    # Functions needed to treat reader as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.join()
