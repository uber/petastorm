from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
from pyspark.sql.session import SparkSession

import atexit
import os
import shutil
import uuid

DEFAULT_CACHE_DIR = "/tmp/spark-converter"
ROW_GROUP_SIZE = 32 * 1024 * 1024


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
        reader = make_batch_reader("file://" + self.cache_file_path)
        return tf_dataset_context_manager(reader)

    def delete(self):
        """
        Delete cache files at self.cache_file_path.
        """
        shutil.rmtree(self.cache_file_path, ignore_errors=True)


class tf_dataset_context_manager:

    def __init__(self, reader):
        """
        :param reader: A :class:`petastorm.reader.Reader` object.
        """
        self.reader = reader
        self.dataset = make_petastorm_dataset(reader)

    def __enter__(self):
        return self.dataset

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reader.stop()
        self.reader.join()


def _cache_df_or_retrieve_cache_path(df, cache_dir, row_group_size):
    """
    Check whether the df is cached.
    If so, return the existing cache file path.
    If not, cache the df into the cache_dir in parquet format and return the cache file path.
    Use atexit to delete the cache before the python interpreter exits.
    :param df:        A :class:`DataFrame` object.
    :param cache_dir: A string denoting the directory for the saved parquet file.
    :return:          A string denoting the path of the saved parquet file.
    """
    uuid_str = str(uuid.uuid4())
    save_to_dir = os.path.join(cache_dir, uuid_str)
    df.write.mode("overwrite") \
        .option("parquet.block.size", row_group_size) \
        .parquet(save_to_dir)
    atexit.register(shutil.rmtree, save_to_dir, True)

    # remove _xxx files, which will break `pyarrow.parquet` loading
    underscore_files = [f for f in os.listdir(save_to_dir) if f.startswith("_")]
    for f in underscore_files:
        os.remove(os.path.join(save_to_dir, f))
    return save_to_dir


def make_spark_converter(df, cache_dir=None, row_group_size=ROW_GROUP_SIZE):
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
    :param row_group_size: An int denoting the number of bytes in a parquet row group.

    :return: a :class:`SparkDatasetConverter` object that holds the materialized dataframe and
            can be used to make one or more tensorflow datasets or torch dataloaders.
    """
    spark = SparkSession.builder.getOrCreate()
    if cache_dir is None:
        cache_dir = spark.conf \
            .get("spark.petastorm.converter.default.cache.dir", DEFAULT_CACHE_DIR)
    cache_file_path = _cache_df_or_retrieve_cache_path(df, cache_dir, row_group_size)
    dataset_size = spark.read.parquet(cache_file_path).count()
    return SparkDatasetConverter(cache_file_path, dataset_size)
