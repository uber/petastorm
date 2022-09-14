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

import collections.abc
import logging
import warnings

import six
from pyarrow import parquet as pq

from petastorm.arrow_reader_worker import ArrowReaderWorker
from petastorm.cache import NullCache
from petastorm.errors import NoDataAvailableError
from petastorm.etl import dataset_metadata, rowgroup_indexing
from petastorm.etl.dataset_metadata import PetastormMetadataError, infer_or_load_unischema
from petastorm.fs_utils import get_filesystem_and_path_or_paths, normalize_dir_url
from petastorm.local_disk_cache import LocalDiskCache
from petastorm.ngram import NGram
from petastorm.predicates import PredicateBase
from petastorm.py_dict_reader_worker import PyDictReaderWorker
from petastorm.reader_impl.arrow_table_serializer import ArrowTableSerializer
from petastorm.reader_impl.pickle_serializer import PickleSerializer
from petastorm.selectors import RowGroupSelectorBase
from petastorm.transform import transform_schema
from petastorm.workers_pool.dummy_pool import DummyPool
from petastorm.workers_pool.process_pool import ProcessPool
from petastorm.workers_pool.thread_pool import ThreadPool
from petastorm.workers_pool.ventilator import ConcurrentVentilator

logger = logging.getLogger(__name__)

# Ventilator guarantees that no more than workers + _VENTILATE_EXTRA_ROWGROUPS are processed at a moment by a
# worker pool. This guarantees that we don't run out of memory if data consumer is slower than the Reader.
_VENTILATE_EXTRA_ROWGROUPS = 2

LOCAL_DISK_CACHE = 'local-disk'
NULL_CACHE = 'null'


def normalize_dataset_url_or_urls(dataset_url_or_urls):
    if isinstance(dataset_url_or_urls, list):
        if not dataset_url_or_urls:
            raise ValueError('dataset url list must be non-empty.')
        return [normalize_dir_url(url) for url in dataset_url_or_urls]
    else:
        return normalize_dir_url(dataset_url_or_urls)


def make_reader(dataset_url,
                schema_fields=None,
                reader_pool_type='thread', workers_count=10, pyarrow_serialize=False, results_queue_size=50,
                seed=None, shuffle_rows=False,
                shuffle_row_groups=True, shuffle_row_drop_partitions=1,
                predicate=None,
                rowgroup_selector=None,
                num_epochs=1,
                cur_shard=None, shard_count=None, shard_seed=None,
                cache_type=NULL_CACHE, cache_location=None, cache_size_limit=None,
                cache_row_size_estimate=None, cache_extra_settings=None,
                hdfs_driver='libhdfs3',
                transform_spec=None,
                filters=None,
                storage_options=None,
                zmq_copy_buffers=True,
                filesystem=None):
    """
    Creates an instance of Reader for reading Petastorm datasets. A Petastorm dataset is a dataset generated using
    :func:`~petastorm.etl.dataset_metadata.materialize_dataset` context manager as explained
    `here <https://petastorm.readthedocs.io/en/latest/readme_include.html#generating-a-dataset>`_.

    See :func:`~petastorm.make_batch_reader` to read from a Parquet store that was not generated using
    :func:`~petastorm.etl.dataset_metadata.materialize_dataset`.

    :param dataset_url: a url to a parquet directory or a url list (with the same scheme) to parquet files.
        e.g. ``'hdfs://some_hdfs_cluster/user/yevgeni/parquet8'``, or ``'file:///tmp/mydataset'``,
        or ``'s3://bucket/mydataset'``, or ``'gs://bucket/mydataset'``,
        or ``[file:///tmp/mydataset/00000.parquet, file:///tmp/mydataset/00001.parquet]``.
    :param schema_fields: Can be: a list of unischema fields and/or regex pattern strings; ``None`` to read all fields;
            an NGram object, then it will return an NGram of the specified fields.
    :param reader_pool_type: A string denoting the reader pool type. Should be one of ['thread', 'process', 'dummy']
        denoting a thread pool, process pool, or running everything in the master thread. Defaults to 'thread'
    :param workers_count: An int for the number of workers to use in the reader pool. This only is used for the
        thread or process pool. Defaults to 10
    :param pyarrow_serialize: THE ARGUMENT IS DEPRECATED AND WILL BE REMOVED IN FUTURE VERSIONS.
    :param results_queue_size: Size of the results queue to store prefetched row-groups. Currently only applicable to
        thread reader pool type.
    :param seed: Random seed specified for shuffle and sharding with reproducible outputs. Defaults to None
    :param shuffle_rows: Whether to shuffle inside a single row group. Defaults to False.
    :param shuffle_row_groups: Whether to shuffle row groups (the order in which full row groups are read)
    :param shuffle_row_drop_partitions: This is is a positive integer which determines how many partitions to
        break up a row group into for increased shuffling in exchange for worse performance (extra reads).
        For example if you specify 2 each row group read will drop half of the rows within every row group and
        read the remaining rows in separate reads. It is recommended to keep this number below the regular row
        group size in order to not waste reads which drop all rows.
    :param predicate: instance of :class:`.PredicateBase` object to filter rows to be returned by reader. The predicate
        will be passed a single row and must return a boolean value indicating whether to include it in the results.
    :param rowgroup_selector: instance of row group selector object to select row groups to be read
    :param num_epochs: An epoch is a single pass over all rows in the dataset. Setting ``num_epochs`` to
        ``None`` will result in an infinite number of epochs.
    :param cur_shard: An int denoting the current shard number. Each node reading a shard should
        pass in a unique shard number in the range [0, shard_count). shard_count must be supplied as well.
        Defaults to None
    :param shard_count: An int denoting the number of shards to break this dataset into. Defaults to None
    :param shard_seed: (Deprecated) Random seed used for sharding row groups. Defaults to None
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
    :param transform_spec: An instance of :class:`~petastorm.transform.TransformSpec` object defining how a record
        is transformed after it is loaded and decoded. The transformation occurs on a worker thread/process (depends
        on the ``reader_pool_type`` value).
    :param filters: (List[Tuple] or List[List[Tuple]]): Standard PyArrow filters.
        These will be applied when loading the parquet file with PyArrow. More information
        here: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
    :param storage_options: Dict of kwargs forwarded to ``fsspec`` to initialize the filesystem.
    :param zmq_copy_buffers: A bool indicating whether to use 0mq copy buffers with ProcessPool.
    :param filesystem: An instance of ``pyarrow.FileSystem`` to use. Will ignore storage_options and
        other filesystem configs if it's provided.
    :return: A :class:`Reader` object
    """
    dataset_url_or_urls = normalize_dataset_url_or_urls(dataset_url)

    filesystem, dataset_path = get_filesystem_and_path_or_paths(
        dataset_url_or_urls,
        hdfs_driver,
        storage_options=storage_options,
        filesystem=filesystem
    )

    if cache_type is None or cache_type == NULL_CACHE:
        cache = NullCache()
    elif cache_type == LOCAL_DISK_CACHE:
        cache = LocalDiskCache(cache_location, cache_size_limit, cache_row_size_estimate, **cache_extra_settings or {})
    else:
        raise ValueError('Unknown cache_type: {}'.format(cache_type))

    try:
        dataset_metadata.get_schema_from_dataset_url(dataset_url_or_urls, hdfs_driver=hdfs_driver,
                                                     storage_options=storage_options, filesystem=filesystem)
    except PetastormMetadataError:
        warnings.warn('Currently make_reader supports reading only Petastorm datasets. '
                      'To read from a non-Petastorm Parquet store use make_batch_reader')

    if reader_pool_type == 'thread':
        reader_pool = ThreadPool(workers_count, results_queue_size)
    elif reader_pool_type == 'process':
        if pyarrow_serialize:
            warnings.warn("pyarrow_serializer was deprecated and will be removed in future versions. "
                          "The argument no longer has any effect.")
        serializer = PickleSerializer()
        reader_pool = ProcessPool(workers_count, serializer, zmq_copy_buffers=zmq_copy_buffers)
    elif reader_pool_type == 'dummy':
        reader_pool = DummyPool()
    else:
        raise ValueError('Unknown reader_pool_type: {}'.format(reader_pool_type))

    kwargs = {
        'schema_fields': schema_fields,
        'reader_pool': reader_pool,
        'shuffle_rows': shuffle_rows,
        'seed': seed,
        'shuffle_row_groups': shuffle_row_groups,
        'shuffle_row_drop_partitions': shuffle_row_drop_partitions,
        'predicate': predicate,
        'rowgroup_selector': rowgroup_selector,
        'num_epochs': num_epochs,
        'cur_shard': cur_shard,
        'shard_count': shard_count,
        'shard_seed': shard_seed,
        'cache': cache,
        'transform_spec': transform_spec,
        'filters': filters
    }

    try:
        return Reader(filesystem, dataset_path,
                      worker_class=PyDictReaderWorker,
                      is_batched_reader=False,
                      **kwargs)
    except PetastormMetadataError as e:
        logger.error('Unexpected exception: %s', str(e))
        raise RuntimeError('make_reader has failed. If you were trying to open a Parquet store that was not '
                           'created using Petastorm materialize_dataset and it contains only scalar columns, '
                           'you may use make_batch_reader to read it.\n'
                           'Inner exception: %s', str(e))


def make_batch_reader(dataset_url_or_urls,
                      schema_fields=None,
                      reader_pool_type='thread', workers_count=10,
                      seed=None, shuffle_rows=False,
                      shuffle_row_groups=True, shuffle_row_drop_partitions=1,
                      predicate=None,
                      rowgroup_selector=None,
                      num_epochs=1,
                      cur_shard=None, shard_count=None, shard_seed=None,
                      cache_type='null', cache_location=None, cache_size_limit=None,
                      cache_row_size_estimate=None, cache_extra_settings=None,
                      hdfs_driver='libhdfs3',
                      transform_spec=None,
                      filters=None,
                      storage_options=None,
                      zmq_copy_buffers=True,
                      filesystem=None):
    """
    Creates an instance of Reader for reading batches out of a non-Petastorm Parquet store.

    Currently, only stores having native scalar parquet data types are supported.
    Use :func:`~petastorm.make_reader` to read Petastorm Parquet stores generated with
    :func:`~petastorm.etl.dataset_metadata.materialize_dataset`.

    NOTE: only scalar columns or array type (of primitive type element) columns are currently supported.

    NOTE: If without `schema_fields` specified, the reader schema will be inferred from parquet dataset. then the
    reader schema fields order will preserve parqeut dataset fields order (partition column come first), but if
    setting `transform_spec` and specified `TransformSpec.selected_fields`, then the reader schema fields order
    will be the order of 'selected_fields'.

    :param dataset_url_or_urls: a url to a parquet directory or a url list (with the same scheme) to parquet files.
        e.g. ``'hdfs://some_hdfs_cluster/user/yevgeni/parquet8'``, or ``'file:///tmp/mydataset'``,
        or ``'s3://bucket/mydataset'``, or ``'gs://bucket/mydataset'``,
        or ``[file:///tmp/mydataset/00000.parquet, file:///tmp/mydataset/00001.parquet]``.
    :param schema_fields: A list of regex pattern strings. Only columns matching at least one of the
        patterns in the list will be loaded.
    :param reader_pool_type: A string denoting the reader pool type. Should be one of ['thread', 'process', 'dummy']
        denoting a thread pool, process pool, or running everything in the master thread. Defaults to 'thread'
    :param workers_count: An int for the number of workers to use in the reader pool. This only is used for the
        thread or process pool. Defaults to 10
    :param seed: Random seed specified for shuffle and sharding with reproducible outputs. Defaults to None
    :param shuffle_rows: Whether to shuffle inside a single row group. Defaults to False.
    :param shuffle_row_groups: Whether to shuffle row groups (the order in which full row groups are read)
    :param shuffle_row_drop_partitions: This is is a positive integer which determines how many partitions to
        break up a row group into for increased shuffling in exchange for worse performance (extra reads).
        For example if you specify 2 each row group read will drop half of the rows within every row group and
        read the remaining rows in separate reads. It is recommended to keep this number below the regular row
        group size in order to not waste reads which drop all rows.
    :param predicate: instance of :class:`.PredicateBase` object to filter rows to be returned by reader. The predicate
        will be passed a pandas DataFrame object and must return a pandas Series with boolean values of matching
        dimensions.
    :param rowgroup_selector: instance of row group selector object to select row groups to be read
    :param num_epochs: An epoch is a single pass over all rows in the dataset. Setting ``num_epochs`` to
        ``None`` will result in an infinite number of epochs.
    :param cur_shard: An int denoting the current shard number. Each node reading a shard should
        pass in a unique shard number in the range [0, shard_count). shard_count must be supplied as well.
        Defaults to None
    :param shard_count: An int denoting the number of shards to break this dataset into. Defaults to None
    :param shard_seed: (Deprecated) Random seed used for sharding row groups. Defaults to None
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
    :param transform_spec: An instance of :class:`~petastorm.transform.TransformSpec` object defining how a record
        is transformed after it is loaded and decoded. The transformation occurs on a worker thread/process (depends
        on the ``reader_pool_type`` value).
    :param filters: (List[Tuple] or List[List[Tuple]]): Standard PyArrow filters.
        These will be applied when loading the parquet file with PyArrow. More information
        here: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
    :param storage_options: Dict of kwargs forwarded to ``fsspec`` to initialize the filesystem.
    :param zmq_copy_buffers: A bool indicating whether to use 0mq copy buffers with ProcessPool.
    :param filesystem: An instance of ``pyarrow.FileSystem`` to use. Will ignore storage_options and
        other filesystem configs if it's provided.
    :return: A :class:`Reader` object
    """
    dataset_url_or_urls = normalize_dataset_url_or_urls(dataset_url_or_urls)

    filesystem, dataset_path_or_paths = get_filesystem_and_path_or_paths(
        dataset_url_or_urls,
        hdfs_driver,
        storage_options=storage_options,
        filesystem=filesystem
    )

    try:
        dataset_metadata.get_schema_from_dataset_url(dataset_url_or_urls, hdfs_driver=hdfs_driver,
                                                     storage_options=storage_options, filesystem=filesystem)
        warnings.warn('Please use make_reader (instead of \'make_batch_dataset\' function to read this dataset. '
                      'You may get unexpected results. '
                      'Currently make_batch_reader supports reading only Parquet stores that contain '
                      'standard Parquet data types and do not require petastorm decoding.')
    except PetastormMetadataError:
        pass

    if cache_type is None or cache_type == NULL_CACHE:
        cache = NullCache()
    elif cache_type == LOCAL_DISK_CACHE:
        cache = LocalDiskCache(cache_location, cache_size_limit, cache_row_size_estimate,
                               **cache_extra_settings or {})
    else:
        raise ValueError('Unknown cache_type: {}'.format(cache_type))

    if reader_pool_type == 'thread':
        reader_pool = ThreadPool(workers_count)
    elif reader_pool_type == 'process':
        serializer = ArrowTableSerializer()
        reader_pool = ProcessPool(workers_count, serializer, zmq_copy_buffers=zmq_copy_buffers)
    elif reader_pool_type == 'dummy':
        reader_pool = DummyPool()
    else:
        raise ValueError('Unknown reader_pool_type: {}'.format(reader_pool_type))

    return Reader(filesystem, dataset_path_or_paths,
                  schema_fields=schema_fields,
                  worker_class=ArrowReaderWorker,
                  reader_pool=reader_pool,
                  seed=seed,
                  shuffle_rows=shuffle_rows,
                  shuffle_row_groups=shuffle_row_groups,
                  shuffle_row_drop_partitions=shuffle_row_drop_partitions,
                  predicate=predicate,
                  rowgroup_selector=rowgroup_selector,
                  num_epochs=num_epochs,
                  cur_shard=cur_shard,
                  shard_count=shard_count,
                  shard_seed=shard_seed,
                  cache=cache,
                  transform_spec=transform_spec,
                  is_batched_reader=True,
                  filters=filters)


class Reader(object):
    """Reads a dataset from a Petastorm dataset.

    :ivar last_row_consumed: True if the last row was already returned by the Reader.
    """

    def __init__(self, pyarrow_filesystem, dataset_path, schema_fields=None,
                 seed=None, shuffle_rows=False, shuffle_row_groups=True,
                 shuffle_row_drop_partitions=1,
                 predicate=None, rowgroup_selector=None, reader_pool=None, num_epochs=1,
                 cur_shard=None, shard_count=None, cache=None, worker_class=None,
                 transform_spec=None, is_batched_reader=False, filters=None, shard_seed=None):
        """Initializes a reader object.

        :param pyarrow_filesystem: An instance of ``pyarrow.FileSystem`` that will be used. If not specified,
            then a default one will be selected based on the url (only for ``hdfs://`` or ``file://``; for
            ``s3://`` and ``gs://`` support, use ``make_reader``). The default hdfs driver is ``libhdfs3``.
            If you want to to use ``libhdfs``, use
            ``pyarrow_filesystem=pyarrow.hdfs.connect('hdfs:///some/path', driver='libhdfs')``.
        :param dataset_path: filepath to a parquet directory or parquet file path list on the specified filesystem.
            e.g. ``'/user/yevgeni/parquet8'``, or ``'/tmp/mydataset'``,
            or ``[/tmp/mydataset/00000.parquet, /tmp/mydataset/00001.parquet]``
        :param schema_fields: Either list of unischema fields to subset, or ``None`` to read all fields.
            OR an NGram object, then it will return an NGram of the specified properties.
        :param seed: Random seed specified for shuffle and sharding with reproducible outputs. Defaults to None
        :param shuffle_rows: Whether to shuffle inside a single row group. Defaults to False.
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
        :param shard_seed: (Deprecated) Random seed used for sharding row groups. Defaults to None
        :param cache: An object conforming to :class:`.CacheBase` interface. Before loading row groups from a parquet
            file the Reader will attempt to load these values from cache. Caching is useful when communication
            to the main data store is either slow or expensive and the local machine has large enough storage
            to store entire dataset (or a partition of a dataset if shards are used).
            By default, use the :class:`.NullCache` implementation.
        :param worker_class: This is the class that will be instantiated on a different thread/process. It's
            responsibility is to load and filter the data.
        :param filters: (List[Tuple] or List[List[Tuple]]): Standard PyArrow filters.
            These will be applied when loading the parquet file with PyArrow. More information
            here: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
        """
        self.num_epochs = num_epochs

        # 1. Open the parquet storage (dataset)
        # 2. Get a list of all groups
        # 3. Filter rowgroups
        #    a. predicates
        #    b. row-group selector (our indexing mechanism)
        #    c. partition: used to get a subset of data for distributed training
        # 4. Create a rowgroup ventilator object
        # 5. Start workers pool
        if not (isinstance(schema_fields, collections.abc.Iterable) or isinstance(schema_fields, NGram)
                or schema_fields is None):
            raise ValueError('Fields must be either None, an iterable collection of Unischema fields '
                             'or an NGram object.')

        self.is_batched_reader = is_batched_reader
        # 1. Resolve dataset path (hdfs://, file://) and open the parquet storage (dataset)
        self.dataset = pq.ParquetDataset(dataset_path, filesystem=pyarrow_filesystem,
                                         validate_schema=False, metadata_nthreads=10,
                                         filters=filters)

        stored_schema = infer_or_load_unischema(self.dataset)

        if isinstance(schema_fields, NGram):
            self.ngram = schema_fields
            self.ngram.resolve_regex_field_names(stored_schema)
        else:
            self.ngram = None

        # By default, use original method of working with list of dictionaries and not arrow tables
        worker_class = worker_class or PyDictReaderWorker
        self._results_queue_reader = worker_class.new_results_queue_reader()

        if self.ngram and not self.ngram.timestamp_overlap and shuffle_row_drop_partitions > 1:
            raise NotImplementedError('Using timestamp_overlap=False is not implemented with'
                                      ' shuffle_options.shuffle_row_drop_partitions > 1')

        cache = cache or NullCache()

        self._workers_pool = reader_pool or ThreadPool(10)

        # Make a schema view (a view is a Unischema containing only a subset of fields
        # Will raise an exception if invalid schema fields are in schema_fields
        if self.ngram:
            fields = self.ngram.get_field_names_at_all_timesteps()
        else:
            fields = schema_fields if isinstance(schema_fields, collections.abc.Iterable) else None

        storage_schema = stored_schema.create_schema_view(fields) if fields else stored_schema
        if len(storage_schema.fields) == 0:
            raise RuntimeError(f"No fields matching the criteria '{fields}' were found in the dataset {dataset_path}.")
        if transform_spec:
            self.schema = transform_schema(storage_schema, transform_spec)
        else:
            self.schema = storage_schema

        # 2. Get a list of all row groups
        row_groups = dataset_metadata.load_row_groups(self.dataset)

        # 3. Filter rowgroups
        _shard_seed = seed
        if shard_seed:
            warnings.warn("shard_seed was deprecated and will be removed in future versions. "
                          "Use seed to apply randomization effects on sharding row groups.")
            # Use shard_seed to overwrite, to be removed in future.
            _shard_seed = shard_seed
        filtered_row_group_indexes, worker_predicate = self._filter_row_groups(self.dataset, row_groups, predicate,
                                                                               rowgroup_selector, cur_shard,
                                                                               shard_count, _shard_seed)
        # 4. Create a rowgroup ventilator object
        normalized_shuffle_row_drop_partitions = \
            self._normalize_shuffle_options(shuffle_row_drop_partitions, self.dataset)
        self.ventilator = self._create_ventilator(filtered_row_group_indexes, shuffle_row_groups,
                                                  normalized_shuffle_row_drop_partitions,
                                                  self.num_epochs, worker_predicate,
                                                  self._workers_pool.workers_count + _VENTILATE_EXTRA_ROWGROUPS,
                                                  seed)

        # 5. Start workers pool
        self._workers_pool.start(worker_class, (pyarrow_filesystem, dataset_path, storage_schema,
                                                self.ngram, row_groups, cache, transform_spec,
                                                self.schema, filters, shuffle_rows, seed),
                                 ventilator=self.ventilator)
        logger.debug('Workers pool started')

        self.last_row_consumed = False
        self.stopped = False

    def reset(self):
        """Resets ``Reader`` state and allows to fetch more samples once the ``Reader`` finished reading all epochs,
        as specified by the ``num_epochs`` parameter.

        Once all samples were read from a reader, an attempt to fetch new sample (e.g. ``next(reader)`` would raise
        ``StopIterationError``. You can reset the reader to the original state and restart reading samples
        calling ``reset()``.

        We do not support calling ``reset()`` until all samples were consumed. ``NotImplementedError``
        will be raised if a user attempt to do so.

        Calling reset after ``stop()`` was called has no effect.

        :return: None
        """
        # TODO(yevgeni): could there be a race here?
        if not self.last_row_consumed:
            # Don't allow reseting in the middle of epoch iterations since it is not very well defined how
            # to treat samples that are already 'in-flight': do we need to stop emitting results immediately and
            # drop these in-flight samples? Or just ignore it? What would happen if we have two concurrent ventilators
            # that are emitting load requests at the same time?
            raise NotImplementedError('Currently do not support resetting a reader while in the middle of iteration. '
                                      'You can call reset only after all samples were consumed.')
        self.last_row_consumed = False
        self.ventilator.reset()

    @property
    def batched_output(self):
        return self._results_queue_reader.batched_output

    def _filter_row_groups(self, dataset, row_groups, predicate, rowgroup_selector, cur_shard,
                           shard_count, seed):
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
        :param shard_count: An int denoting the number of reader shards
        :param seed: If not None: random seed to shuffle row groups for data sharding.
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
                                                                    filtered_row_group_indexes, seed)

        if not filtered_row_group_indexes:
            warnings.warn('No matching data is available for loading after rowgroup '
                          'selector were applied and the data was sharded.')

        return filtered_row_group_indexes, worker_predicate

    def _partition_row_groups(self, dataset, row_groups, shard_count, cur_shard,
                              filtered_row_group_indexes, seed):
        """Filters the list of row group indexes based on the requested training partitions. Returns
        a modified list of rowgroup indexes."""

        if not shard_count \
                or not isinstance(cur_shard, int) \
                or not isinstance(shard_count, int):
            raise ValueError('partition and num_partitions must be ints and both specified to use partitioning')

        if shard_count is not None and len(row_groups) < shard_count:
            raise NoDataAvailableError('Number of row-groups in the dataset must be greater or equal to the number of '
                                       'requested shards. Otherwise, some of the shards will end up being empty.')

        # We hash on the relative path of each parquet file to guarantee consistency between different reader
        # constructions even after moving the dataset
        if seed is not None:
            import random
            # Instantiate a new Random class with a seed and sample from it.
            # Avoid affecting global/default random generator.
            shard_random = random.Random(seed)
            shard_random.shuffle(filtered_row_group_indexes)

        filtered_row_group_indexes = [index for index in filtered_row_group_indexes if index % shard_count == cur_shard]
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

            partition_names = dataset.partitions.partition_names if dataset.partitions else set()
            if set(predicate_fields) == partition_names:
                assert len(partition_names) == 1, \
                    'Datasets with only a single partition level supported at the moment'

                filtered_row_group_indexes = []
                for piece_index, piece in enumerate(row_groups):
                    partition_name, partition_index = piece.partition_keys[0]
                    partition_value = dataset.partitions[0].keys[partition_index]

                    # Convert partition value to correct type per the schema
                    partition_value = self.schema.fields[partition_name].numpy_dtype(partition_value)
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
                           num_epochs, worker_predicate, max_ventilation_queue_size, seed):
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
                                    randomize_item_order=shuffle_row_groups,
                                    random_seed=seed)

    def stop(self):
        """Stops all worker threads/processes."""
        self._workers_pool.stop()
        self.stopped = True

    def join(self):
        """Joins all worker threads/processes. Will block until all worker workers have been fully terminated."""
        self._workers_pool.join()

    @property
    def diagnostics(self):
        return self._workers_pool.diagnostics

    def __iter__(self):
        return self

    def __next__(self):
        if self.stopped:
            raise RuntimeError('Trying to read a sample after a reader created by '
                               'make_reader/make_batch_reader has stopped. This may happen if the '
                               'make_reader/make_batch_reader context manager has exited but you try to '
                               'fetch a sample from it anyway')
        try:
            return self._results_queue_reader.read_next(self._workers_pool, self.schema, self.ngram)
        except StopIteration:
            self.last_row_consumed = True
            raise

    def next(self):
        return self.__next__()

    # Functions needed to treat reader as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.join()
