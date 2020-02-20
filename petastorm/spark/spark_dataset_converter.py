from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
from pyspark.sql.dataframe import DataFrame

import numpy as np
import os
import tensorflow as tf

assert(tf.version.VERSION == '1.15.0')

# Config
CACHE_DIR = "/tmp/tf"

# primitive type mapping
type_map = {
    "boolean": np.bool_,
    "byte": np.int8,  # 8-bit signed
    "double": np.float64,
    "float": np.float32,
    "integer": np.int32,
    "long": np.int64,
    "short": np.int16
}


class tf_dataset_context_manager:

    def __init__(self, converter):
        self.converter = converter

    def __enter__(self) -> tf.data.Dataset:
        return self.converter.dataset

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.converter.close()


class SparkDatasetConverter:
    """
    The SparkDatasetConverter class manages the intermediate files when converting a SparkDataFrame to a
    Tensorflow Dataset or PyTorch DataLoader.
    """
    def __init__(self, cache_file_path):
        """
        :param cache_file_path: The path to store the cache files.
        """
        self.cache_file_path = cache_file_path
        self.dataset = None

    def make_tf_dataset(self) -> tf_dataset_context_manager:
        reader = make_batch_reader("file://" + self.cache_file_path)
        self.dataset = make_petastorm_dataset(reader)
        return tf_dataset_context_manager(self)

    def close(self):
        """
        Todo: delete cache files
        :return:
        """
        pass


def _cache_df_or_retrieve_cache_path(df: DataFrame, dir_path: str) -> str:
    """
    Check whether the df is cached.
    If so, return the existing cache file path.
    If not, cache the df into the configured cache dir in parquet format and return the cache file path.
    :param df: SparkDataFrame
    :param dir_path: the directory for the saved parquet file, could be local, hdfs, dbfs, ...
    :return: the path of the saved parquet file
    """
    # Todo: add cache management
    df.write.mode("overwrite") \
        .option("parquet.block.size", 1024 * 1024) \
        .parquet(CACHE_DIR)

    # remove _xxx files, which will break `pyarrow.parquet` loading
    underscore_files = [f for f in os.listdir(dir_path) if f.startswith("_")]
    for f in underscore_files:
        os.remove(os.path.join(dir_path, f))
    return CACHE_DIR


def make_spark_converter(df: DataFrame) -> SparkDatasetConverter:
    """
    This class will check whether the df is cached.
    If so, it will use the existing cache file path to construct a SparkDatasetConverter.
    If not, Materialize the df into the configured cache dir in parquet format and use the cache file path to
    construct a SparkDatasetConverter.
    :param df: The DataFrame to materialize.
    :return: a SparkDatasetConverter
    """
    cache_file_path = _cache_df_or_retrieve_cache_path(df, CACHE_DIR)
    return SparkDatasetConverter(cache_file_path)
