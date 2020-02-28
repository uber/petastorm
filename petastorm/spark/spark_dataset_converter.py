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

from pyarrow import LocalFileSystem
from pyspark.sql.session import SparkSession
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


class SparkDatasetConverter(object):
    """
    A `SparkDatasetConverter` object holds one materialized spark dataframe and
    can be used to make one or more tensorflow datasets or torch dataloaders.
    The `SparkDatasetConverter` object is picklable and can be used in remote
    processes.
    See `make_spark_converter`
    """

    def __init__(self, cache_file_path, dataset_size):
        """
        :param cache_file_path: A string denoting the path to store the cache
            files.
        :param dataset_size: An int denoting the number of rows in the
            dataframe.
        """
        self.cache_file_path = cache_file_path
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def make_tf_dataset(self):
        return _tf_dataset_context_manager(self.cache_file_path)

    def delete(self):
        """
        Delete cache files at self.cache_file_path.
        """
        _delete_cache_data(self.cache_file_path)


class _tf_dataset_context_manager(object):
    """
    A context manager that manages the creation and termination of a
    :class:`petastorm.Reader`.
    """

    def __init__(self, data_url):
        """
        :param data_url: A string specifying the data URL.
        """
        from petastorm.tf_utils import make_petastorm_dataset

        self.reader = make_batch_reader(data_url)
        self.dataset = make_petastorm_dataset(self.reader)

    def __enter__(self):
        return self.dataset

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reader.stop()
        self.reader.join()


def _get_df_plan(df):
    return df._jdf.queryExecution().analyzed()


class CachedDataFrameMeta(object):

    def __init__(self, df, row_group_size, compression_codec):
        self.row_group_size = row_group_size
        self.compression_codec = compression_codec
        # Note: the metadata will hold dataframe plan, but it won't
        # hold the dataframe object (dataframe plan will not reference
        # dataframe object),
        # This means the dataframe can be released by spark gc.
        self.df_plan = _get_df_plan(df)
        self.data_path = None

    @classmethod
    def create_cached_dataframe(cls, df, parent_cache_dir, row_group_size,
                                compression_codec):
        meta = cls(df, row_group_size, compression_codec)
        meta.data_path = _materialize_df(
            df, parent_cache_dir, row_group_size, compression_codec)
        return meta


_cache_df_meta_list = []
_cache_df_meta_list_lock = threading.Lock()


def _cache_df_or_retrieve_cache_path(df, parent_cache_dir,
                                     parquet_row_group_size_bytes,
                                     compression_codec):
    """
    Check whether the df is cached.
    If so, return the existing cache file path.
    If not, cache the df into the cache_dir in parquet format and return the
    cache file path.
    Use atexit to delete the cache before the python interpreter exits.
    :param df: A :class:`DataFrame` object.
    :param parent_cache_dir: A string denoting the directory for the saved
        parquet file.
    :param parquet_row_group_size_bytes: An int denoting the number of bytes
        in a parquet row group.
    :param compression_codec: Specify compression codec.
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
                    meta.df_plan.sameResult(df_plan):
                return meta.data_path
        # do not find cached dataframe, start materializing.
        cached_df_meta = CachedDataFrameMeta.create_cached_dataframe(
            df, parent_cache_dir, parquet_row_group_size_bytes,
            compression_codec)
        _cache_df_meta_list.append(cached_df_meta)
        return cached_df_meta.data_path


def _materialize_df(df, parent_cache_dir, parquet_row_group_size_bytes,
                    compression_codec):
    uuid_str = str(uuid.uuid4())
    save_to_dir = os.path.join(parent_cache_dir, uuid_str)

    df.write \
        .option("compression", compression_codec) \
        .option("parquet.block.size", parquet_row_group_size_bytes) \
        .parquet(save_to_dir)
    atexit.register(_delete_cache_data, save_to_dir)

    return save_to_dir


def make_spark_converter(
        df,
        cache_dir_url=None,
        parquet_row_group_size_bytes=DEFAULT_ROW_GROUP_SIZE_BYTES,
        compression=None):
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

    :return: a :class:`SparkDatasetConverter` object that holds the
        materialized dataframe and can be used to make one or more tensorflow
        datasets or torch dataloaders.
    """
    if cache_dir_url is None:
        cache_dir_url = _get_spark_session().conf \
            .get("petastorm.spark.converter.defaultCacheDirUrl", None)

    scheme = urlparse(cache_dir_url).scheme
    if not scheme:
        raise ValueError(
            'ERROR! A scheme-less dataset url ({}) is no longer supported. '
            'Please prepend "file://" for local filesystem.'.format(scheme))

    if cache_dir_url is None:
        raise ValueError(
            "Please specify the parameter cache_dir_url denoting the parent "
            "directory to store intermediate files, or set the spark config "
            "`petastorm.spark.converter.defaultCacheDirUrl`.")

    if compression is None:
        # TODO: Improve default behavior to be automatically choosing the
        #  best way.
        compression_codec = "uncompressed"
    elif compression:
        compression_codec = "snappy"
    else:
        compression_codec = "uncompressed"

    cache_file_path = _cache_df_or_retrieve_cache_path(
        df, cache_dir_url, parquet_row_group_size_bytes, compression_codec)
    dataset_size = _get_spark_session().read.parquet(cache_file_path).count()
    return SparkDatasetConverter(cache_file_path, dataset_size)
