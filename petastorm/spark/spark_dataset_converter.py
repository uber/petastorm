from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
from pyspark.sql.session import SparkSession

import atexit
import os
import shutil
import threading
import uuid

DEFAULT_CACHE_DIR = "/tmp/spark-converter"
ROW_GROUP_SIZE = 32 * 1024 * 1024


def _get_spark_session():
    return SparkSession.builder.getOrCreate()


class SparkDatasetConverter(object):
    """
    A `SparkDatasetConverter` object holds one materialized spark dataframe and
    can be used to make one or more tensorflow datasets or torch dataloaders.
    The `SparkDatasetConverter` object is picklable and can be used in remote processes.
    See `make_spark_converter`
    """
    def __init__(self, cache_file_path, dataset_size):
        """
        :param cache_file_path: A string denoting the path to store the cache files.
        :param dataset_size: An int denoting the number of rows in the dataframe.
        """
        self.cache_file_path = cache_file_path
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def make_tf_dataset(self):
        # TODO: make data_uri support both local fs and hdfs
        #   1. if cache_file_path is local path, convert it into "file:///..."
        #   2. if cache_file_path is hdfs path: "hdfs:/...", keep it unchanged
        #   3. if other cases, raise error.
        data_uri = "file://" + self.cache_file_path
        return tf_dataset_context_manager(data_uri)

    def delete(self):
        """
        Delete cache files at self.cache_file_path.
        """
        # TODO:
        #   make it support both local fs and hdfs
        shutil.rmtree(self.cache_file_path, ignore_errors=True)


class tf_dataset_context_manager:

    def __init__(self, data_uri):
        """
        :param reader: A :class:`petastorm.reader.Reader` object.
        """
        self.reader = make_batch_reader(data_uri)
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
        # hold the dataframe object (dataframe plan will not reference dataframe object),
        # This means the dataframe can be released by spark gc.
        self.df_plan = _get_df_plan(df)
        self.data_path = None

    @classmethod
    def create_cached_dataframe(cls, df, parent_cache_dir, row_group_size, compression_codec):
        meta = cls(df, row_group_size, compression_codec)
        meta.data_path = _materialize_df(
            df, parent_cache_dir, row_group_size, compression_codec)
        return meta


_cache_df_meta_list = []
_cache_df_meta_list_lock = threading.Lock()


def _cache_df_or_retrieve_cache_path(df, parent_cache_dir, row_group_size, compression_codec):
    """
    Check whether the df is cached.
    If so, return the existing cache file path.
    If not, cache the df into the cache_dir in parquet format and return the cache file path.
    Use atexit to delete the cache before the python interpreter exits.
    :param df:        A :class:`DataFrame` object.
    :param parent_cache_dir: A string denoting the directory for the saved parquet file.
    :param compression_codec: Specify compression codec.
    :return:          A string denoting the path of the saved parquet file.
    """
    # TODO
    #  Improve the cache list by hash table (Note we need use hash(df_plan + row_group_size)
    with _cache_df_meta_list_lock:
        df_plan = _get_df_plan(df)
        for meta in _cache_df_meta_list:
            if meta.row_group_size == row_group_size and \
                    meta.compression_codec == compression_codec and \
                    meta.df_plan.sameResult(df_plan):
                return meta.data_path
        # do not find cached dataframe, start materializing.
        cached_df_meta = CachedDataFrameMeta.create_cached_dataframe(
            df, parent_cache_dir, row_group_size, compression_codec)
        _cache_df_meta_list.append(cached_df_meta)
        return cached_df_meta.data_path


def _materialize_df(df, parent_cache_dir, row_group_size, compression_codec):
    uuid_str = str(uuid.uuid4())
    save_to_dir = os.path.join(parent_cache_dir, uuid_str)

    df.write \
        .option("compression", compression_codec) \
        .option("parquet.block.size", row_group_size) \
        .parquet(save_to_dir)

    # TODO: support both local fs and hdfs
    atexit.register(shutil.rmtree, save_to_dir, True)

    return save_to_dir


def make_spark_converter(
        df,
        cache_dir=None,
        parquet_row_group_size=ROW_GROUP_SIZE,
        compression=None):
    """
    Convert a spark dataframe into a :class:`SparkDatasetConverter` object. It will materialize
    a spark dataframe to a `cache_dir` or a default cache directory.
    The returned `SparkDatasetConverter` object will hold the materialized dataframe, and
    can be used to make one or more tensorflow datasets or torch dataloaders.

    :param df:        The :class:`DataFrame` object to be converted.
    :param cache_dir: A string denoting the parent directory to store intermediate files.
                      Default None, it will fallback to the spark config
                      "spark.petastorm.converter.default.cache.dir".
                      If the spark config is empty, it will fallback to DEFAULT_CACHE_DIR.
    :param compression: True or False, specify whether to apply compression. Default None.
                        If None, will automatically choose the best way.
    :param parquet_row_group_size: An int denoting the number of bytes in a parquet row group.

    :return: a :class:`SparkDatasetConverter` object that holds the materialized dataframe and
            can be used to make one or more tensorflow datasets or torch dataloaders.
    """
    if cache_dir is None:
        cache_dir = _get_spark_session().conf \
            .get("spark.petastorm.converter.default.cache.dir", DEFAULT_CACHE_DIR)

    if compression is None:
        # TODO: Improve default behavior to be automatically choosing the best way.
        compression_codec = "uncompressed"
    elif compression:
        compression_codec = "snappy"
    else:
        compression_codec = "uncompressed"

    cache_file_path = _cache_df_or_retrieve_cache_path(
        df, cache_dir, parquet_row_group_size, compression_codec)
    dataset_size = _get_spark_session().read.parquet(cache_file_path).count()
    return SparkDatasetConverter(cache_file_path, dataset_size)
