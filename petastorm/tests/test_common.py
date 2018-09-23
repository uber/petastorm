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

from __future__ import division

from decimal import Decimal
from functools import partial

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, ShortType, LongType, DecimalType

from petastorm.codecs import CompressedImageCodec, NdarrayCodec, \
    ScalarCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.etl.rowgroup_indexers import SingleFieldIndexer
from petastorm.etl.rowgroup_indexing import build_rowgroup_index
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row

_DEFAULT_IMAGE_SIZE = (32, 16, 3)

TestSchema = Unischema('TestSchema', [
    UnischemaField('partition_key', np.unicode_, (), ScalarCodec(StringType()), False),
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
    UnischemaField('sensor_name', np.unicode_, (1,), NdarrayCodec(), False),
    UnischemaField('string_array_nullable', np.unicode_, (None,), NdarrayCodec(), True),
])


def _randomize_row(id_num):
    """Returns a row with random values"""
    row_dict = {
        TestSchema.id.name: id_num,
        TestSchema.id2.name: id_num % 2,
        TestSchema.partition_key.name: 'p_{}'.format(int(id_num / 10)),
        TestSchema.python_primitive_uint8.name: np.random.randint(0, 255),
        TestSchema.image_png.name: np.random.randint(0, 255, _DEFAULT_IMAGE_SIZE).astype(np.uint8),
        TestSchema.matrix.name: np.random.randint(0, 255, _DEFAULT_IMAGE_SIZE).astype(np.float32),
        TestSchema.decimal.name: Decimal(np.random.randint(0, 255) / Decimal(100)),
        TestSchema.matrix_uint16.name: np.random.randint(0, 255, _DEFAULT_IMAGE_SIZE).astype(np.uint16),
        TestSchema.matrix_string.name: np.random.randint(0, 100, (4,)).astype(np.string_),
        TestSchema.empty_matrix_string.name: np.asarray([], dtype=np.string_),
        TestSchema.matrix_nullable.name: None,
        TestSchema.sensor_name.name: np.asarray(['test_sensor'], dtype=np.unicode_),
        TestSchema.string_array_nullable.name:
            None if id_num % 5 == 0 else np.asarray([], dtype=np.unicode_)
            if id_num % 4 == 0 else np.asarray([str(i + id_num) for i in range(2)], dtype=np.unicode_),
    }
    return row_dict


def create_test_dataset(tmp_url, rows, num_files=2, spark=None):
    """
    Creates a test dataset under tmp_dir, with rows and num_files that has TestSchema.
    :param tmp_url: The URL of the temp directory to store the test dataset in.
    :param rows: The number of rows for the dataset.
    :param num_files: The number of files to partition the data between.
    :param spark: An optional spark session to use
    :return: A list of the dataset dictionary.
    """

    shutdown = False
    if not spark:
        spark_session = SparkSession \
            .builder \
            .appName('petastorm_end_to_end_test') \
            .master('local[8]')

        spark = spark_session.getOrCreate()
        shutdown = True
    spark_context = spark.sparkContext

    with materialize_dataset(spark, tmp_url, TestSchema):
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

    # Create list of objects to build row group indexes
    indexers = [
        SingleFieldIndexer(TestSchema.id.name, TestSchema.id.name),
        SingleFieldIndexer(TestSchema.sensor_name.name, TestSchema.sensor_name.name),
        SingleFieldIndexer(TestSchema.string_array_nullable.name, TestSchema.string_array_nullable.name),
    ]
    build_rowgroup_index(tmp_url, spark_context, indexers)

    if shutdown:
        spark.stop()

    return dataset_dicts
