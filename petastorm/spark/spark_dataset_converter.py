#  Copyright (c) 2020 Databricks, Inc.
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

import atexit
import datetime
import logging
import os
import shutil
import threading
import time
import uuid
from distutils.version import LooseVersion
from multiprocessing.pool import ThreadPool
from typing import List, Any

import pyspark
from pyarrow import LocalFileSystem
from pyspark.sql.session import SparkSession
from pyspark.sql.types import ArrayType, DoubleType, FloatType
from six.moves.urllib.parse import urlparse

from petastorm import make_batch_reader
from petastorm.fs_utils import (FilesystemResolver,
                                get_filesystem_and_path_or_paths, normalize_dir_url)
from fsspec.core import strip_protocol

if LooseVersion(pyspark.__version__) < LooseVersion('3.0'):
    def vector_to_array(_1, _2='float32'):
        raise RuntimeError("Vector columns are only supported in pyspark>=3.0")
else:
    from pyspark.ml.functions import vector_to_array  # type: ignore  # pylint: disable=import-error

DEFAULT_ROW_GROUP_SIZE_BYTES = 32 * 1024 * 1024

logger = logging.getLogger(__name__)


def _get_spark_session():
    """Get or create spark session. Note: This function can only be invoked from driver side."""
    if pyspark.TaskContext.get() is not None:
        # This is a safety check.
        raise RuntimeError('_get_spark_session should not be invoked from executor side.')
    return SparkSession.builder.getOrCreate()


_parent_cache_dir_url = None


def _get_parent_cache_dir_url():
    """Get parent cache dir url from `petastorm.spark.converter.parentCacheDirUrl`
    We can only set the url config once.
    """
    global _parent_cache_dir_url  # pylint: disable=global-statement

    conf_url = _get_spark_session().conf \
        .get(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, None)

    if conf_url is None:
        raise ValueError(
            "Please set the spark config {}.".format(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF))

    conf_url = normalize_dir_url(conf_url)
    _check_parent_cache_dir_url(conf_url)
    _parent_cache_dir_url = conf_url
    logger.info(
        'Read %s %s', SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, _parent_cache_dir_url)

    return _parent_cache_dir_url


def _default_delete_dir_handler(dataset_url):
    resolver = FilesystemResolver(dataset_url)
    fs = resolver.filesystem()
    _dataset_url = strip_protocol(dataset_url)

    if isinstance(fs, LocalFileSystem):
        # pyarrow has a bug: LocalFileSystem.delete() is not implemented.
        # https://issues.apache.org/jira/browse/ARROW-7953
        # We can remove this branch once ARROW-7953 is fixed.
        local_path = _dataset_url
        if os.path.exists(local_path):
            shutil.rmtree(local_path, ignore_errors=False)
    else:
        if fs.exists(_dataset_url):
            fs.delete(_dataset_url, recursive=True)


_delete_dir_handler = _default_delete_dir_handler


def register_delete_dir_handler(handler):
    """Register a handler for delete a directory url.

    :param handler: A deleting function which take a argument of directory url.
                    If ``None``, use the default handler, note the default handler
                    will use libhdfs3 driver.

    """
    global _delete_dir_handler  # pylint: disable=global-statement
    if handler is None:
        _delete_dir_handler = _default_delete_dir_handler
    else:
        _delete_dir_handler = handler


def _delete_cache_data_atexit(dataset_url):
    try:
        _delete_dir_handler(dataset_url)
    except Exception as e:  # pylint: disable=broad-except
        logger.warning('Delete cache data %s failed due to %s', dataset_url, repr(e))


def _get_horovod_rank_and_size():
    """Get rank and size from environment, return (rank, size), if failed, return (``None``, ``None``)"""
    rank_env = ['HOROVOD_RANK', 'OMPI_COMM_WORLD_RANK', 'PMI_RANK']
    size_env = ['HOROVOD_SIZE', 'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE']

    for rank_var, size_var in zip(rank_env, size_env):
        rank = os.environ.get(rank_var)
        size = os.environ.get(size_var)
        if rank is not None and size is not None:
            return int(rank), int(size)
        elif rank is not None or size is not None:
            return None, None

    return None, None


def _check_rank_and_size_consistent_with_horovod(petastorm_reader_kwargs):
    """Check whether the ``cur_shard`` and ``shard_count`` args are consistent with horovod environment variables.

    If not consistent with horovod environment variables, log warning message and return ``False``.
    If there're no related horovod environment variable set, return ``True``.
    """
    hvd_rank, hvd_size = _get_horovod_rank_and_size()
    cur_shard = petastorm_reader_kwargs.get('cur_shard')
    shard_count = petastorm_reader_kwargs.get('shard_count')

    if hvd_rank is not None and hvd_size is not None:
        if cur_shard != hvd_rank or shard_count != hvd_size:
            logger.warning(
                'The petastorm reader arguments cur_shard(%d) and '
                'shard_count(%d) is not consistent with horovod '
                'environments hvd_rank(%d) and hvd_size(%d), If you want '
                'each horovod worker train on one corresponding shard data, '
                'you should set argument `cur_shard` to be `hvd.rank()` '
                'and argument `shard_count` to be `hvd.size()`.',
                cur_shard, shard_count, hvd_rank, hvd_size)
            return False
    return True


class SparkDatasetConverter(object):
    """A `SparkDatasetConverter` object holds one materialized spark dataframe and
    can be used to make one or more tensorflow datasets or torch dataloaders.
    The `SparkDatasetConverter` object is picklable and can be used in remote
    processes.
    See `make_spark_converter`
    """

    PARENT_CACHE_DIR_URL_CONF = 'petastorm.spark.converter.parentCacheDirUrl'

    def __init__(self, cache_dir_url, file_urls, dataset_size):
        """
        :param cache_dir_url: A string denoting the path to store the cache files.
        :param file_urls: a list of parquet file url list of this dataset.
        :param dataset_size: An int denoting the number of rows in the dataframe.
        """
        self.cache_dir_url = cache_dir_url
        self.file_urls = file_urls
        self.dataset_size = dataset_size

    def __len__(self):
        """
        :return: dataset size
        """
        return self.dataset_size

    @staticmethod
    def _check_and_set_overriden_petastorm_args(petastorm_reader_kwargs, num_epochs, workers_count):
        # override some arguments default values of petastorm reader
        petastorm_reader_kwargs['num_epochs'] = num_epochs
        if workers_count is None:
            # TODO: generate a best tuned value for default worker count value
            workers_count = 4
        petastorm_reader_kwargs['workers_count'] = workers_count
        _check_rank_and_size_consistent_with_horovod(petastorm_reader_kwargs)

    def make_tf_dataset(
            self,
            batch_size=None,
            prefetch=None,
            num_epochs=None,
            workers_count=None,
            shuffling_queue_capacity=None,
            **petastorm_reader_kwargs
    ):
        """Make a tensorflow dataset.

        This method will do the following two steps:
          1) Open a petastorm reader on the materialized dataset dir.
          2) Create a tensorflow dataset based on the reader created in (1)

        :param batch_size: The number of items to return per batch. Default ``None``.
            If ``None``, current implementation will set batch size to be 32, in future,
            ``None`` value will denotes auto tuned best value for batch size.
        :param prefetch: Prefetch size for tensorflow dataset. If ``None`` will use
            tensorflow autotune size. Note only available on tensorflow>=1.14
        :param num_epochs: An epoch is a single pass over all rows in the dataset.
            Setting ``num_epochs`` to ``None`` will result in an infinite number
            of epochs.
        :param workers_count: An int for the number of workers to use in the
            reader pool. This only is used for the thread or process pool.
            ``None`` denotes auto tune best value (current implementation when auto tune,
            it will always use 4 workers, but it may be improved in future)
            Default value ``None``.
        :param shuffling_queue_capacity: An int specifying the number of items to fill into a queue
            from which items are sampled each step to form batches. The larger the capacity, the
            better shuffling of the elements within the dataset. The default value of ``None``
            results in no shuffling.
        :param petastorm_reader_kwargs: arguments for `petastorm.make_batch_reader()`,
            exclude these arguments: ``dataset_url``, ``num_epochs``, ``workers_count``.

        :return: a context manager for a `tf.data.Dataset` object.
                 when exit the returned context manager, the reader
                 will be closed.
        """
        self._check_and_set_overriden_petastorm_args(
            petastorm_reader_kwargs, num_epochs=num_epochs, workers_count=workers_count)
        return TFDatasetContextManager(
            self.file_urls,
            batch_size=batch_size,
            prefetch=prefetch,
            petastorm_reader_kwargs=petastorm_reader_kwargs,
            shuffling_queue_capacity=shuffling_queue_capacity)

    def make_torch_dataloader(self,
                              batch_size=32,
                              num_epochs=None,
                              workers_count=None,
                              shuffling_queue_capacity=0,
                              data_loader_fn=None,
                              **petastorm_reader_kwargs):
        """Make a PyTorch DataLoader.

        This method will do the following two steps:
          1) Open a petastorm reader on the materialized dataset dir.
          2) Create a PyTorch DataLoader based on the reader created in (1)

        :param batch_size: The number of items to return per batch. Default ``None``.
            If ``None``, current implementation will set batch size to be 32, in future,
            ``None`` value will denotes auto tuned best value for batch size.
        :param num_epochs: An epoch is a single pass over all rows in the
            dataset. Setting ``num_epochs`` to ``None`` will result in an
            infinite number of epochs.
        :param workers_count: An int for the number of workers to use in the
            reader pool. This only is used for the thread or process pool.
            Defaults value ``None``, which means using the default value from
            `petastorm.make_batch_reader()`. We can autotune it in the future.
        :param shuffling_queue_capacity: Queue capacity is passed to the underlying
            :class:`tf.RandomShuffleQueue` instance. If set to 0, no shuffling will be done.
        :param data_loader_fn: A function (or class) that generates a
            `torch.utils.data.DataLoader` object. The default value of ``None`` uses
            `petastorm.pytorch.DataLoader`.
        :param petastorm_reader_kwargs: arguments for `petastorm.make_batch_reader()`,
            exclude these arguments: ``dataset_url``, ``num_epochs``, ``workers_count``.

        :return: a context manager for a `torch.utils.data.DataLoader` object.
                 when exit the returned context manager, the reader
                 will be closed.
        """
        self._check_and_set_overriden_petastorm_args(
            petastorm_reader_kwargs, num_epochs=num_epochs, workers_count=workers_count)
        return TorchDatasetContextManager(
            self.file_urls,
            batch_size=batch_size,
            petastorm_reader_kwargs=petastorm_reader_kwargs,
            shuffling_queue_capacity=shuffling_queue_capacity,
            data_loader_fn=data_loader_fn)

    def delete(self):
        """Delete cache files at self.cache_dir_url."""
        _remove_cache_metadata_and_data(self.cache_dir_url)


class TFDatasetContextManager(object):
    """A context manager that manages the creation and termination of a
    :class:`petastorm.Reader`.
    """

    def __init__(
            self,
            parquet_file_url_list,
            batch_size,
            prefetch,
            petastorm_reader_kwargs,
            shuffling_queue_capacity,
    ):
        """
        :param parquet_file_url_list: A string specifying the parquet file URL list.
        :param batch_size: batch size for tensorflow dataset.
        :param prefetch: the prefectch size for tensorflow dataset.
        :param petastorm_reader_kwargs: other arguments for petastorm reader
        :param shuffling_queue_capacity: the shuffle queue capacity for the tensorflow dataset
        """
        self.parquet_file_url_list = parquet_file_url_list
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.petastorm_reader_kwargs = petastorm_reader_kwargs
        self.shuffling_queue_capacity = shuffling_queue_capacity

    def __enter__(self):
        # import locally to avoid importing tensorflow globally.
        from petastorm.tf_utils import make_petastorm_dataset
        import tensorflow.compat.v1 as tf  # pylint: disable=import-error

        _wait_file_available(self.parquet_file_url_list)

        self.reader = make_batch_reader(self.parquet_file_url_list, **self.petastorm_reader_kwargs)

        # unroll dataset
        dataset = make_petastorm_dataset(self.reader).flat_map(
            tf.data.Dataset.from_tensor_slices)

        if self.shuffling_queue_capacity:
            dataset = dataset.shuffle(self.shuffling_queue_capacity)

        # TODO: auto tune best batch size in default case.
        batch_size = self.batch_size or 32
        dataset = dataset.batch(batch_size=batch_size)

        prefetch = self.prefetch

        if prefetch is None:
            if LooseVersion(tf.__version__) >= LooseVersion('1.14'):
                # We can make prefetch optimization
                prefetch = tf.data.experimental.AUTOTUNE
            else:
                prefetch = 1

        dataset = dataset.prefetch(prefetch)

        return dataset

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reader.stop()
        self.reader.join()


class TorchDatasetContextManager(object):
    """A context manager that manages the creation and termination of a
    :class:`petastorm.Reader`.
    """

    def __init__(self,
                 parquet_file_url_list,
                 batch_size,
                 petastorm_reader_kwargs,
                 shuffling_queue_capacity,
                 data_loader_fn):
        """
        :param parquet_file_url_list: A string specifying the parquet file URL list.
        :param batch_size: The number of items to return per batch. Default ``None``.
            If ``None``, current implementation will set batch size to be 32, in future,
            ``None`` value will denotes auto tuned best value for batch size.
        :param petastorm_reader_kwargs: other arguments for petastorm reader
        :param shuffling_queue_capacity: Queue capacity is passed to the underlying
            :class:`tf.RandomShuffleQueue` instance. If set to 0, no shuffling will be done.
        :param data_loader_fn: function to generate the PyTorch DataLoader.

        See `SparkDatasetConverter.make_torch_dataloader()`  for the definitions
        of the other parameters.
        """
        self.parquet_file_url_list = parquet_file_url_list
        self.batch_size = batch_size
        self.petastorm_reader_kwargs = petastorm_reader_kwargs
        self.shuffling_queue_capacity = shuffling_queue_capacity
        self.data_loader_fn = data_loader_fn

    def __enter__(self):
        from petastorm.pytorch import DataLoader

        _wait_file_available(self.parquet_file_url_list)

        self.reader = make_batch_reader(self.parquet_file_url_list, **self.petastorm_reader_kwargs)

        data_loader_fn = self.data_loader_fn or DataLoader
        self.loader = data_loader_fn(reader=self.reader,
                                     batch_size=self.batch_size,
                                     shuffling_queue_capacity=self.shuffling_queue_capacity)
        return self.loader

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reader.stop()
        self.reader.join()


def _get_df_plan(df):
    return df._jdf.queryExecution().analyzed()


class CachedDataFrameMeta(object):

    def __init__(self, df, parent_cache_dir_url, row_group_size, compression_codec, dtype):
        self.row_group_size = row_group_size
        self.compression_codec = compression_codec
        # Note: the metadata will hold dataframe plan, but it won't
        # hold the dataframe object (dataframe plan will not reference
        # dataframe object),
        # This means the dataframe can be released by spark gc.
        self.df_plan = _get_df_plan(df)
        self.cache_dir_url = None
        self.dtype = dtype
        self.parent_cache_dir_url = parent_cache_dir_url

    @classmethod
    def create_cached_dataframe_meta(cls, df, parent_cache_dir_url, row_group_size,
                                     compression_codec, dtype):
        meta = cls(df, parent_cache_dir_url, row_group_size, compression_codec, dtype)
        meta.cache_dir_url = _materialize_df(
            df,
            parent_cache_dir_url=parent_cache_dir_url,
            parquet_row_group_size_bytes=row_group_size,
            compression_codec=compression_codec,
            dtype=dtype)
        return meta


_cache_df_meta_list: List[Any] = []  # TODO(Yevgeni): can be more precise with the type (instead of Any)
_cache_df_meta_list_lock = threading.Lock()


def _is_spark_local_mode():
    return _get_spark_session().conf.get('spark.master').strip().lower().startswith('local')


def _check_url(dir_url):
    """Check dir url, will check scheme, raise error if empty scheme"""
    parsed = urlparse(dir_url)
    if not parsed.scheme:
        raise ValueError(
            'ERROR! A scheme-less directory url ({}) is no longer supported. '
            'Please prepend "file://" for local filesystem.'.format(dir_url))


def _check_parent_cache_dir_url(dir_url):
    """Check dir url whether is suitable to be used as parent cache directory."""
    _check_url(dir_url)
    fs, dir_path = get_filesystem_and_path_or_paths(dir_url)
    if 'DATABRICKS_RUNTIME_VERSION' in os.environ and not _is_spark_local_mode():
        if isinstance(fs, LocalFileSystem):
            # User need to use dbfs fuse URL.
            if not dir_path.startswith('/dbfs/'):
                logger.warning(
                    "Usually, when running on databricks spark cluster, you should specify a dbfs fuse path "
                    "for %s, like: 'file:/dbfs/path/to/cache_dir', otherwise, you should mount NFS to this "
                    "directory '%s' on all nodes of the cluster, e.g. using EFS.",
                    SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, dir_url)


def _make_sub_dir_url(dir_url, name):
    parsed = urlparse(dir_url)
    new_path = parsed.path + '/' + name
    return parsed._replace(path=new_path).geturl()


def _cache_df_or_retrieve_cache_data_url(df, parent_cache_dir_url,
                                         parquet_row_group_size_bytes,
                                         compression_codec,
                                         dtype):
    """Check whether the df is cached.

    If so, return the existing cache file path.
    If not, cache the df into the cache_dir in parquet format and return the
    cache file path.
    Use atexit to delete the cache before the python interpreter exits.
    :param df: A :class:`DataFrame` object.
    :param parquet_row_group_size_bytes: An int denoting the number of bytes
        in a parquet row group.
    :param compression_codec: Specify compression codec.
    :param dtype: ``None``, 'float32' or 'float64', specifying the precision of the floating-point
        elements in the output dataset. Integer types will remain unchanged. If ``None``, all types
        will remain unchanged. Default 'float32'.
    :return: A string denoting the path of the saved parquet file.
    """
    # TODO
    #  Improve the cache list by hash table (Note we need use hash(df_plan +
    #  row_group_size)
    with _cache_df_meta_list_lock:
        df_plan = _get_df_plan(df)
        for meta in _cache_df_meta_list:
            if meta.row_group_size == parquet_row_group_size_bytes and \
                    meta.compression_codec == compression_codec and \
                    meta.df_plan.sameResult(df_plan) and \
                    meta.dtype == dtype and \
                    meta.parent_cache_dir_url == parent_cache_dir_url:
                return meta.cache_dir_url
        # do not find cached dataframe, start materializing.
        cached_df_meta = CachedDataFrameMeta.create_cached_dataframe_meta(
            df, parent_cache_dir_url, parquet_row_group_size_bytes,
            compression_codec, dtype)
        _cache_df_meta_list.append(cached_df_meta)
        return cached_df_meta.cache_dir_url


def _remove_cache_metadata_and_data(cache_dir_url):
    with _cache_df_meta_list_lock:
        for i in range(len(_cache_df_meta_list)):
            if _cache_df_meta_list[i].cache_dir_url == cache_dir_url:
                _cache_df_meta_list.pop(i)
                break
    _delete_dir_handler(cache_dir_url)


def _convert_precision(df, dtype):
    if dtype is None:
        return df

    if dtype != "float32" and dtype != "float64":
        raise ValueError("dtype {} is not supported. \
            Use 'float32' or float64".format(dtype))

    source_type, target_type = (DoubleType, FloatType) \
        if dtype == "float32" else (FloatType, DoubleType)

    logger.warning("Converting floating-point columns to %s", dtype)

    for field in df.schema:
        col_name = field.name
        if isinstance(field.dataType, source_type):
            df = df.withColumn(col_name, df[col_name].cast(target_type()))
        elif isinstance(field.dataType, ArrayType) and \
                isinstance(field.dataType.elementType, source_type):
            df = df.withColumn(col_name, df[col_name].cast(ArrayType(target_type())))
    return df


def _convert_vector(df, dtype):
    from pyspark.ml.linalg import VectorUDT
    from pyspark.mllib.linalg import VectorUDT as OldVectorUDT

    for field in df.schema:
        col_name = field.name
        if isinstance(field.dataType, VectorUDT) or \
                isinstance(field.dataType, OldVectorUDT):
            df = df.withColumn(col_name,
                               vector_to_array(df[col_name], dtype))
    return df


def _gen_cache_dir_name():
    """Generate a random directory name for storing dataset.
    The directory name format is:
      {datetime}-{spark_application_id}-{uuid4}
    This will help user to find the related spark application for a directory.
    So that if atexit deletion failed, user can manually delete them.
    """
    uuid_str = str(uuid.uuid4())
    time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    appid = _get_spark_session().sparkContext.applicationId
    return '{time}-appid-{appid}-{uuid}'.format(time=time_str, appid=appid, uuid=uuid_str)


def _materialize_df(df, parent_cache_dir_url, parquet_row_group_size_bytes,
                    compression_codec, dtype):
    dir_name = _gen_cache_dir_name()
    save_to_dir_url = _make_sub_dir_url(parent_cache_dir_url, dir_name)
    df = _convert_vector(df, dtype)
    df = _convert_precision(df, dtype)

    df.write \
        .option("compression", compression_codec) \
        .option("parquet.block.size", parquet_row_group_size_bytes) \
        .parquet(save_to_dir_url)

    logger.info('Materialize dataframe to url %s successfully.', save_to_dir_url)

    atexit.register(_delete_cache_data_atexit, save_to_dir_url)

    return save_to_dir_url


_FILE_AVAILABILITY_WAIT_TIMEOUT_SECS = 30


def _wait_file_available(url_list):
    """Waiting about _FILE_AVAILABILITY_WAIT_TIMEOUT_SECS seconds (default 30 seconds) to make sure
    all files are available for reading. This is useful in some filesystems, such as S3 which only
    providing eventually consistency.
    """
    fs, path_list = get_filesystem_and_path_or_paths(url_list)
    logger.debug('Waiting some seconds until all parquet-store files appear at urls %s', ','.join(url_list))

    def wait_for_file(path):
        end_time = time.time() + _FILE_AVAILABILITY_WAIT_TIMEOUT_SECS
        while time.time() < end_time:
            if fs.exists(path):
                return True
            time.sleep(0.1)
        return False

    pool = ThreadPool(64)
    try:
        results = pool.map(wait_for_file, path_list)
        failed_list = [url for url, result in zip(url_list, results) if not result]
        if failed_list:
            raise RuntimeError('Timeout while waiting for all parquet-store files to appear at urls {failed_list},'
                               'Please check whether these files were saved successfully when materializing dataframe.'
                               .format(failed_list=','.join(failed_list)))
    finally:
        pool.close()
        pool.join()


def _check_dataset_file_median_size(url_list):
    fs, path_list = get_filesystem_and_path_or_paths(url_list)
    RECOMMENDED_FILE_SIZE_BYTES = 50 * 1024 * 1024

    # TODO: also check file size for other file system.
    if isinstance(fs, LocalFileSystem):
        pool = ThreadPool(64)
        try:
            file_size_list = pool.map(os.path.getsize, path_list)
            if len(file_size_list) > 1:
                mid_index = len(file_size_list) // 2
                median_size = sorted(file_size_list)[mid_index]  # take the larger one if tie
                if median_size < RECOMMENDED_FILE_SIZE_BYTES:
                    logger.warning('The median size %d B (< 50 MB) of the parquet files is too small. '
                                   'Total size: %d B. Increase the median file size by calling df.repartition(n) or '
                                   'df.coalesce(n), which might help improve the performance. Parquet files: %s, ...',
                                   median_size, sum(file_size_list), url_list[0])
        finally:
            pool.close()
            pool.join()


def make_spark_converter(
        df,
        parquet_row_group_size_bytes=DEFAULT_ROW_GROUP_SIZE_BYTES,
        compression_codec=None,
        dtype='float32'
):
    """Convert a spark dataframe into a :class:`SparkDatasetConverter` object.
    It will materialize a spark dataframe to the directory specified by
    spark conf 'petastorm.spark.converter.parentCacheDirUrl'.
    The dataframe will be materialized in parquet format, and we can specify
    `parquet_row_group_size_bytes` and `compression_codec` for the parquet
    format. See params documentation for details.

    The returned `SparkDatasetConverter` object will hold the materialized
    dataframe, and can be used to make one or more tensorflow datasets or
    torch dataloaders.

    We can explicitly delete the materialized dataframe data, see
    `SparkDatasetConverter.delete`, and when the spark application exit,
    it will try best effort to delete the materialized dataframe data.

    :param df: The :class:`DataFrame` object to be converted.
    :param parquet_row_group_size_bytes: An int denoting the number of bytes
        in a parquet row group when materializing the dataframe.
    :param compression_codec: Specify compression codec.
        It can be one of 'uncompressed', 'bzip2', 'gzip', 'lz4', 'snappy', 'deflate'.
        Default ``None``. If ``None``, it will leave the data uncompressed.
    :param dtype: ``None``, 'float32' or 'float64', specifying the precision of the floating-point
        elements in the output dataset. Integer types will remain unchanged. If ``None``, all types
        will remain unchanged. Default 'float32'.

    :return: a :class:`SparkDatasetConverter` object that holds the
        materialized dataframe and can be used to make one or more tensorflow
        datasets or torch dataloaders.
    """

    parent_cache_dir_url = _get_parent_cache_dir_url()

    # TODO: Improve default behavior to be automatically choosing the best way.
    compression_codec = compression_codec or "uncompressed"

    if compression_codec.lower() not in \
            ['uncompressed', 'bzip2', 'gzip', 'lz4', 'snappy', 'deflate']:
        raise RuntimeError(
            "compression_codec should be None or one of the following values: "
            "'uncompressed', 'bzip2', 'gzip', 'lz4', 'snappy', 'deflate'")

    dataset_cache_dir_url = _cache_df_or_retrieve_cache_data_url(
        df, parent_cache_dir_url, parquet_row_group_size_bytes, compression_codec, dtype)

    # TODO: improve this by read parquet file metadata to get count
    #  Currently spark can make sure to only read the minimal column
    #  so count will usually be fast.
    spark = _get_spark_session()
    spark_df = spark.read.parquet(dataset_cache_dir_url)

    dataset_size = spark_df.count()
    parquet_file_url_list = list(spark_df._jdf.inputFiles())
    _check_dataset_file_median_size(parquet_file_url_list)

    return SparkDatasetConverter(dataset_cache_dir_url, parquet_file_url_list, dataset_size)
