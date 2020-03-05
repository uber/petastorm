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
import os
import shutil
import threading
import uuid
import warnings

from distutils.version import LooseVersion

from pyarrow import LocalFileSystem
from pyspark.sql.session import SparkSession
from pyspark.sql.types import FloatType, DoubleType, ArrayType
from six.moves.urllib.parse import urlparse

from petastorm import make_batch_reader
from petastorm.fs_utils import FilesystemResolver

DEFAULT_ROW_GROUP_SIZE_BYTES = 32 * 1024 * 1024


def _get_spark_session():
    return SparkSession.builder.getOrCreate()


def _delete_cache_data(dataset_url):
    """
    Delete the cache data in the underlying file system.
    """
    resolver = FilesystemResolver(dataset_url)
    fs = resolver.filesystem()
    parsed = urlparse(dataset_url)
    if isinstance(fs, LocalFileSystem):
        # pyarrow has a bug: LocalFileSystem.delete() is not implemented.
        # https://issues.apache.org/jira/browse/ARROW-7953
        # We can remove this branch once ARROW-7953 is fixed.
        local_path = parsed.path
        if os.path.exists(local_path):
            shutil.rmtree(local_path, ignore_errors=False)
    else:
        if fs.exists(dataset_url):
            fs.delete(dataset_url, recursive=True)


def _delete_cache_data_atexit(dataset_url):
    try:
        _delete_cache_data(dataset_url)
    except BaseException:
        warnings.warn('delete cache data {url} failed.'.format(url=dataset_url))


class SparkDatasetConverter(object):
    """
    A `SparkDatasetConverter` object holds one materialized spark dataframe and
    can be used to make one or more tensorflow datasets or torch dataloaders.
    The `SparkDatasetConverter` object is picklable and can be used in remote
    processes.
    See `make_spark_converter`
    """

    def __init__(self, cache_dir_url, dataset_size):
        """
        :param cache_dir_url: A string denoting the path to store the cache
            files.
        :param dataset_size: An int denoting the number of rows in the
            dataframe.
        """
        self.cache_dir_url = cache_dir_url
        self.dataset_size = dataset_size

    def __len__(self):
        """
        :return: dataset size
        """
        return self.dataset_size

    def make_tf_dataset(self,
                        batch_size=32,
                        prefetch=None,
                        preproc_fn=None,
                        preproc_parallelism=None):
        """
        Make a tensorflow dataset.

        This method will do the following two steps:
          1) Open a petastorm reader on the materialized dataset dir.
          2) Create a tensorflow dataset based on the reader created in (1)

        :param batch_size: batch size of the generated tf.data.dataset
        :param prefetch: prefetch for tf dataset, if None, will use autotune prefetch
                         if available, if 0, disable prefetch. Default is None.
        :param preproc_fn: preprocessing function, will apply on batched tf tensor.
        :param preproc_parallelism: parallelism for preprocessing function.
                                    If None, will autotune best parallelism if available.
                                    If tf do not support autotune, fallback to 1.

        :return: a context manager for a `tf.data.Dataset` object.
                 when exit the returned context manager, the reader
                 will be closed.
        """
        return _tf_dataset_context_manager(
            self.cache_dir_url,
            batch_size=batch_size,
            prefetch=prefetch,
            preproc_fn=preproc_fn,
            preproc_parallelism=preproc_parallelism
        )

    def delete(self):
        """
        Delete cache files at self.cache_dir_url.
        """
        _delete_cache_data(self.cache_dir_url)


class _tf_dataset_context_manager(object):
    """
    A context manager that manages the creation and termination of a
    :class:`petastorm.Reader`.
    """

    def __init__(self,
                 data_url,
                 batch_size,
                 prefetch,
                 preproc_fn,
                 preproc_parallelism):
        """
        :param data_url: A string specifying the data URL.
        :param batch_size: batch size of the generated tf.data.dataset
        :param prefetch: prefetch for tf dataset
        :param preproc_fn: preprocessing function
        :param preproc_parallelism: parallelism for preprocessing function
        """
        from petastorm.tf_utils import make_petastorm_dataset
        import tensorflow as tf

        def support_prefetch_and_autotune():
            return LooseVersion(tf.__version__) >= LooseVersion('1.14')

        self.reader = make_batch_reader(data_url)
        self.dataset = make_petastorm_dataset(self.reader) \
            .flat_map(tf.data.Dataset.from_tensor_slices) \

        self.dataset = self.dataset.batch(batch_size=batch_size)

        if support_prefetch_and_autotune():
            if prefetch is None:
                prefetch = tf.data.experimental.AUTOTUNE
            if prefetch != 0:
                self.dataset = self.dataset.prefetch(prefetch)

        if preproc_fn is not None:
            if preproc_parallelism is None:
                if support_prefetch_and_autotune():
                    preproc_parallelism = tf.data.experimental.AUTOTUNE
                else:
                    preproc_parallelism = 1
            self.dataset = self.dataset.map(preproc_fn, preproc_parallelism)

    def __enter__(self):
        return self.dataset

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reader.stop()
        self.reader.join()


def _get_df_plan(df):
    return df._jdf.queryExecution().analyzed()


class CachedDataFrameMeta(object):

    def __init__(self, df, row_group_size, compression_codec, precision):
        self.row_group_size = row_group_size
        self.compression_codec = compression_codec
        # Note: the metadata will hold dataframe plan, but it won't
        # hold the dataframe object (dataframe plan will not reference
        # dataframe object),
        # This means the dataframe can be released by spark gc.
        self.df_plan = _get_df_plan(df)
        self.cache_dir_url = None
        self.precision = precision

    @classmethod
    def create_cached_dataframe(cls, df, parent_cache_dir_url, row_group_size,
                                compression_codec, precision):
        meta = cls(df,
                   row_group_size=row_group_size,
                   compression_codec=compression_codec,
                   precision=precision)
        meta.cache_dir_url = _materialize_df(
            df,
            parent_cache_dir=parent_cache_dir_url,
            parquet_row_group_size_bytes=row_group_size,
            compression_codec=compression_codec,
            precision=precision)
        return meta


_cache_df_meta_list = []
_cache_df_meta_list_lock = threading.Lock()


def _normalize_dir_url(dir_url):
    """
    Normalize dir url, will do:
    * check scheme, raise error if empty scheme
    * convert the path to be abspath, remove redundant '/' and trailing '/' in path
    """
    parsed = urlparse(dir_url)
    if not parsed.scheme:
        raise ValueError(
            'ERROR! A scheme-less directory url ({}) is no longer supported. '
            'Please prepend "file://" for local filesystem.'.format(dir_url))
    new_parsed = parsed._replace(path=os.path.abspath(parsed.path))
    return new_parsed.geturl()


def _is_sub_dir_url(dir_url1, dir_url2):
    """
    Check whether url1 is a sub directory of url2
    """
    url1 = _normalize_dir_url(dir_url1)
    url2 = _normalize_dir_url(dir_url2)

    parsed1 = urlparse(url1)
    parsed2 = urlparse(url2)

    return parsed1.scheme == parsed2.scheme and \
        parsed1.netloc == parsed2.netloc and \
        parsed1.path.startswith(parsed2.path + os.sep)


def _cache_df_or_retrieve_cache_data_url(
        df, parent_cache_dir_url, parquet_row_group_size_bytes,
        compression_codec, precision):
    """
    Check whether the df is cached.
    If so, return the existing cache file path.
    If not, cache the df into the cache_dir in parquet format and return the
    cache file path.
    Use atexit to delete the cache before the python interpreter exits.
    :param df: A :class:`DataFrame` object.
    :param parent_cache_dir_url: A string denoting the directory for the saved
        parquet file.
    :param parquet_row_group_size_bytes: An int denoting the number of bytes
        in a parquet row group.
    :param compression_codec: Specify compression codec.
    :param precision: 'float32' or 'float64', specifying the precision of the
        output dataset.
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
                    meta.precision == precision and \
                    _is_sub_dir_url(meta.cache_dir_url, parent_cache_dir_url):
                return meta.cache_dir_url
        # do not find cached dataframe, start materializing.
        cached_df_meta = CachedDataFrameMeta.create_cached_dataframe(
            df, parent_cache_dir_url, parquet_row_group_size_bytes,
            compression_codec, precision)
        _cache_df_meta_list.append(cached_df_meta)
        return cached_df_meta.cache_dir_url


def _convert_precision(df, precision):
    if precision != "float32" and precision != "float64":
        raise ValueError("precision {} is not supported. \
            Use 'float32' or float64".format(precision))

    source_type, target_type = (DoubleType, FloatType) \
        if precision == "float32" else (FloatType, DoubleType)

    for struct_field in df.schema:
        col_name = struct_field.name
        if isinstance(struct_field.dataType, source_type):
            df = df.withColumn(col_name, df[col_name].cast(target_type()))
        elif isinstance(struct_field.dataType, ArrayType) and \
                isinstance(struct_field.dataType.elementType, source_type):
            df = df.withColumn(col_name, df[col_name].cast(ArrayType(target_type())))
    return df


def _materialize_df(df, parent_cache_dir, parquet_row_group_size_bytes,
                    compression_codec, precision):
    uuid_str = str(uuid.uuid4())
    save_to_dir = os.path.join(parent_cache_dir, uuid_str)
    df = _convert_precision(df, precision)

    df.write \
        .option("compression", compression_codec) \
        .option("parquet.block.size", parquet_row_group_size_bytes) \
        .parquet(save_to_dir)
    atexit.register(_delete_cache_data_atexit, save_to_dir)

    return save_to_dir


def make_spark_converter(
        df,
        cache_dir_url=None,
        parquet_row_group_size_bytes=DEFAULT_ROW_GROUP_SIZE_BYTES,
        compression=None,
        precision='float32'):
    """
    Convert a spark dataframe into a :class:`SparkDatasetConverter` object.
    It will materialize a spark dataframe to a `cache_dir_url`.
    The returned `SparkDatasetConverter` object will hold the materialized
    dataframe, and can be used to make one or more tensorflow datasets or
    torch dataloaders.

    :param df: The :class:`DataFrame` object to be converted.
    :param cache_dir_url: A URL string denoting the parent directory to store
        intermediate files. Default None, it will fallback to the spark config
        "petastorm.spark.converter.defaultCacheDirUrl".
    :param parquet_row_group_size_bytes: An int denoting the number of bytes
        in a parquet row group.
    :param compression: True or False, specify whether to apply compression.
        Default None. If None, it will automatically choose the best way.
    :param precision: 'float32' or 'float64', specifying the precision of the
        output dataset.

    :return: a :class:`SparkDatasetConverter` object that holds the
        materialized dataframe and can be used to make one or more tensorflow
        datasets or torch dataloaders.
    """
    if cache_dir_url is None:
        cache_dir_url = _get_spark_session().conf \
            .get("petastorm.spark.converter.defaultCacheDirUrl", None)

    if cache_dir_url is None:
        raise ValueError(
            "Please specify the parameter cache_dir_url denoting the parent "
            "directory to store intermediate files, or set the spark config "
            "`petastorm.spark.converter.defaultCacheDirUrl`.")

    cache_dir_url = _normalize_dir_url(cache_dir_url)

    if compression is None:
        # TODO: Improve default behavior to be automatically choosing the
        #  best way.
        compression_codec = "uncompressed"
    elif compression:
        compression_codec = "snappy"
    else:
        compression_codec = "uncompressed"

    dataset_cache_dir_url = _cache_df_or_retrieve_cache_data_url(
        df, cache_dir_url, parquet_row_group_size_bytes, compression_codec,
        precision)
    dataset_size = _get_spark_session().read.parquet(dataset_cache_dir_url).count()
    return SparkDatasetConverter(dataset_cache_dir_url, dataset_size)
