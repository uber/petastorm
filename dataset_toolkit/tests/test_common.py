#
# Uber, Inc. (c) 2018
#
import glob
import os
from decimal import Decimal
from functools import partial

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, ShortType, LongType, DecimalType

from dataset_toolkit.codecs import CompressedImageCodec, NdarrayCodec, \
    ScalarCodec
from dataset_toolkit.etl.dataset_metadata import add_dataset_metadata
from dataset_toolkit.etl.rowgroup_indexers import SingleFieldIndexer
from dataset_toolkit.etl.rowgroup_indexing import build_rowgroup_index
from dataset_toolkit.unischema import Unischema, UnischemaField, dict_to_spark_row

_DEFAULT_IMAGE_SIZE = (32, 16, 3)

TestSchema = Unischema('TestSchema', [
    UnischemaField('partition_key', np.string_, (), ScalarCodec(StringType()), False),
    UnischemaField('id', np.int64, (), ScalarCodec(LongType()), False),
    UnischemaField('id2', np.int32, (), ScalarCodec(ShortType()), False),
    UnischemaField('python_primitive_uint8', np.uint8, (), ScalarCodec(ShortType()), False),
    UnischemaField('image_png', np.uint8, _DEFAULT_IMAGE_SIZE, CompressedImageCodec('png'), False),
    UnischemaField('matrix', np.float32, _DEFAULT_IMAGE_SIZE, NdarrayCodec(), False),
    UnischemaField('decimal', Decimal, (), ScalarCodec(DecimalType(10, 9)), False),
    UnischemaField('matrix_uint16', np.uint16, _DEFAULT_IMAGE_SIZE, NdarrayCodec(), False),
    UnischemaField('matrix_string', np.string_, (None,), NdarrayCodec(), False),
    UnischemaField('empty_matrix_string', np.string_, (None,), NdarrayCodec(), False),
    UnischemaField('matrix_nullable', np.uint16, _DEFAULT_IMAGE_SIZE, NdarrayCodec(), True),
    UnischemaField('sensor_name', np.string_, (1,), NdarrayCodec(), False),
])


def _randomize_row(id):
    """Returns a row with random values"""
    row_dict = {
        TestSchema.id.name: id,
        TestSchema.id2.name: id % 2,
        TestSchema.partition_key.name: 'p_{}'.format(int(id / 100)),
        TestSchema.python_primitive_uint8.name: np.random.randint(0, 255),
        TestSchema.image_png.name: np.random.randint(0, 255, _DEFAULT_IMAGE_SIZE).astype(np.uint8),
        TestSchema.matrix.name: np.random.randint(0, 255, _DEFAULT_IMAGE_SIZE).astype(np.float32),
        TestSchema.decimal.name: Decimal(np.random.randint(0, 255) / Decimal(100)),
        TestSchema.matrix_uint16.name: np.random.randint(0, 255, _DEFAULT_IMAGE_SIZE).astype(np.uint16),
        TestSchema.matrix_string.name: np.random.randint(0, 100, (4,)).astype(np.string_),
        TestSchema.empty_matrix_string.name: np.asarray([], dtype=np.string_),
        TestSchema.matrix_nullable.name: None,
        TestSchema.sensor_name.name: np.asarray(['test_sensor'], dtype=np.string_),
    }
    return row_dict


def create_test_dataset(tmp_url, rows, num_files=2):
    """
    Creates a test dataset under tmp_dir, with rows and num_files that has TestSchema.
    :param tmp_url: The URL of the temp directory to store the test dataset in.
    :param rows: The number of rows for the dataset.
    :param num_files: The number of files to partition the data between.
    :return: A list of the dataset dictionary.
    """
    spark_session = SparkSession \
        .builder \
        .appName('dataset_toolkit_end_to_end_test') \
        .master('local[8]')

    spark = spark_session.getOrCreate()
    spark_context = spark.sparkContext
    hadoop_config = spark_context._jsc.hadoopConfiguration()
    # This results in spark not writing _SUCCESS file. pyarrow does not handle this extra file in a parquet
    # directory correctly
    hadoop_config.set('mapreduce.fileoutputcommitter.marksuccessfuljobs', 'false')

    id_rdd = spark_context.parallelize(rows, numSlices=40)

    # Make up some random data and store it for referencing in the tests
    random_dicts_rdd = id_rdd.map(_randomize_row).cache()
    dataset_dicts = random_dicts_rdd.collect()

    random_rows_rdd = random_dicts_rdd.map(partial(dict_to_spark_row, TestSchema))

    # Create a spark dataframe with the random rows
    dataframe = spark. \
        createDataFrame(random_rows_rdd, TestSchema.as_spark_schema()).sort('id')

    # Save a parquet
    dataframe. \
        coalesce(num_files). \
        write.option('compression', 'none'). \
        partitionBy('partition_key'). \
        mode('overwrite'). \
        parquet(tmp_url)

    add_dataset_metadata(tmp_url, spark_context, TestSchema)

    # Create list of objects to build row group indexes
    indexers = [
        SingleFieldIndexer(TestSchema.id.name, TestSchema.id.name),
        SingleFieldIndexer(TestSchema.sensor_name.name, TestSchema.sensor_name.name)
    ]
    build_rowgroup_index(tmp_url, spark_context, indexers)

    spark_context.stop()

    return dataset_dicts
