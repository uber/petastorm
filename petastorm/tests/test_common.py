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

import random
from collections import OrderedDict
from decimal import Decimal
from functools import partial

import numpy as np
import pyarrow as pa
import pytz
from pyspark import Row
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, DecimalType, DoubleType, StructField, \
    IntegerType, StructType, DateType, TimestampType, ShortType, ArrayType

from petastorm.codecs import CompressedImageCodec, NdarrayCodec, \
    ScalarCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.etl.rowgroup_indexers import SingleFieldIndexer
from petastorm.etl.rowgroup_indexing import build_rowgroup_index
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row

_DEFAULT_IMAGE_SIZE = (32, 16, 3)

TestSchema = Unischema('TestSchema', [
    UnischemaField('partition_key', np.unicode_, ()),
    UnischemaField('id', np.int64, ()),
    UnischemaField('id2', np.int32, (), ScalarCodec(ShortType()), False),  # Explicit scalar codec in some scalar fields
    UnischemaField('id_float', np.float64, ()),
    UnischemaField('id_odd', np.bool_, ()),
    UnischemaField('python_primitive_uint8', np.uint8, ()),
    UnischemaField('image_png', np.uint8, _DEFAULT_IMAGE_SIZE, CompressedImageCodec('png'), False),
    UnischemaField('matrix', np.float32, _DEFAULT_IMAGE_SIZE, NdarrayCodec(), False),
    UnischemaField('decimal', Decimal, (), ScalarCodec(DecimalType(10, 9)), False),
    UnischemaField('matrix_uint16', np.uint16, _DEFAULT_IMAGE_SIZE, NdarrayCodec(), False),
    UnischemaField('matrix_uint32', np.uint32, _DEFAULT_IMAGE_SIZE, NdarrayCodec(), False),
    UnischemaField('matrix_string', np.string_, (None, None,), NdarrayCodec(), False),
    UnischemaField('empty_matrix_string', np.string_, (None,), NdarrayCodec(), False),
    UnischemaField('matrix_nullable', np.uint16, _DEFAULT_IMAGE_SIZE, NdarrayCodec(), True),
    UnischemaField('sensor_name', np.unicode_, (1,), NdarrayCodec(), False),
    UnischemaField('string_array_nullable', np.unicode_, (None,), NdarrayCodec(), True),
    UnischemaField('integer_nullable', np.int32, (), nullable=True),
])


def _random_binary_string_gen(max_length):
    """Returns a single random string up to max_length specified length that may include \x00 character anywhere in the
    string"""
    size = random.randint(0, max_length)
    return ''.join(random.choice(('\x00', 'A', 'B')) for _ in range(size))


def _random_binary_string_matrix(rows, cols, max_length):
    """Returns a list of lists of random strings"""
    return [[_random_binary_string_gen(max_length) for _ in range(cols)] for _ in range(rows)]


def _randomize_row(id_num):
    """Returns a row with random values"""
    row_dict = {
        TestSchema.id.name: np.int64(id_num),
        TestSchema.id2.name: np.int32(id_num % 2),
        TestSchema.id_float.name: np.float64(id_num),
        TestSchema.id_odd.name: np.bool_(id_num % 2),
        TestSchema.partition_key.name: np.unicode_('p_{}'.format(int(id_num / 10))),
        TestSchema.python_primitive_uint8.name: np.random.randint(0, 255, dtype=np.uint8),
        TestSchema.image_png.name: np.random.randint(0, 255, _DEFAULT_IMAGE_SIZE).astype(np.uint8),
        TestSchema.matrix.name: np.random.random(size=_DEFAULT_IMAGE_SIZE).astype(np.float32),
        TestSchema.decimal.name: Decimal(np.random.randint(0, 255) / Decimal(100)),
        TestSchema.matrix_uint16.name: np.random.randint(0, 2 ** 16 - 1, _DEFAULT_IMAGE_SIZE).astype(np.uint16),
        TestSchema.matrix_uint32.name: np.random.randint(0, 2 ** 32 - 1, _DEFAULT_IMAGE_SIZE).astype(np.uint32),
        TestSchema.matrix_string.name: np.asarray(_random_binary_string_matrix(2, 3, 10)).astype(np.bytes_),
        TestSchema.empty_matrix_string.name: np.asarray([], dtype=np.string_),
        TestSchema.matrix_nullable.name: None,
        TestSchema.sensor_name.name: np.asarray(['test_sensor'], dtype=np.unicode_),
        TestSchema.string_array_nullable.name:
            None if id_num % 5 == 0 else np.asarray([], dtype=np.unicode_)
            if id_num % 4 == 0 else np.asarray([str(i + id_num) for i in range(2)], dtype=np.unicode_),
        TestSchema.integer_nullable.name: None if id_num % 2 else np.int32(id_num),
    }
    return row_dict


def create_test_dataset(tmp_url, rows, num_files=2, spark=None, use_summary_metadata=False):
    """
    Creates a test dataset under tmp_dir, with rows and num_files that has TestSchema.
    :param tmp_url: The URL of the temp directory to store the test dataset in.
    :param rows: The number of rows for the dataset.
    :param num_files: The number of files to partition the data between.
    :param spark: An optional spark session to use
    :param use_summary_metadata: If True, _metadata file will be created.
    :return: A list of the dataset dictionary.
    """

    shutdown = False
    if not spark:
        spark_session = SparkSession \
            .builder \
            .appName('petastorm_end_to_end_test') \
            .master('local[*]')

        spark = spark_session.getOrCreate()
        shutdown = True
    spark_context = spark.sparkContext

    with materialize_dataset(spark, tmp_url, TestSchema, use_summary_metadata=use_summary_metadata):
        id_rdd = spark_context.parallelize(rows, numSlices=40)

        # Make up some random data and store it for referencing in the tests
        random_dicts_rdd = id_rdd.map(_randomize_row).cache()
        dataset_dicts = random_dicts_rdd.collect()

        def _partition_key_to_str(row):
            row['partition_key'] = str(row['partition_key'])
            return row

        random_dicts_rdd = random_dicts_rdd.map(_partition_key_to_str)

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
        SingleFieldIndexer(TestSchema.partition_key.name, TestSchema.partition_key.name),
    ]
    build_rowgroup_index(tmp_url, spark_context, indexers)

    if shutdown:
        spark.stop()

    return dataset_dicts


def create_test_scalar_dataset(output_url, num_rows, num_files=4, spark=None, partition_by=None):
    """Creates a dataset in tmp_url location. The dataset emulates non-petastorm dataset, i.e. contains only native
    parquet types.

    These are the fields with mock data:
      'id', 'int_fixed_size_list', 'datetime', 'timestamp', 'string', 'string2', 'float64'

    :param output_url: Url specifying the location the parquet store is written to
    :param num_rows: Number of rows in the generated dataset
    :param num_files: Number of parquet files that will be written into the parquet store
    :param spark: An instance of spark session object. If `None` (default), a new spark session is created.
    :param partition_by: A list of fields to partition the parquet store by.
    :return: A list of records with a copy of the data written to the dataset.
    """

    is_list_of_scalar_broken = pa.__version__ == '0.15.0'

    partition_by = partition_by or []
    shutdown = False
    if not spark:
        spark_session = SparkSession \
            .builder \
            .appName('petastorm_end_to_end_test') \
            .master('local[*]')

        spark = spark_session.getOrCreate()
        shutdown = True
        hadoop_config = spark.sparkContext._jsc.hadoopConfiguration()
        hadoop_config.setInt('parquet.block.size', 100)

    def expected_row(i):
        result = {'id': np.int32(i),
                  'id_div_700': np.int32(i // 700),
                  'datetime': np.datetime64('2019-01-02'),
                  'timestamp': np.datetime64('2005-02-25T03:30'),
                  'string': np.unicode_('hello_{}'.format(i)),
                  'string2': np.unicode_('world_{}'.format(i)),
                  'float64': np.float64(i) * .66}
        if not is_list_of_scalar_broken:
            result['int_fixed_size_list'] = np.arange(1 + i, 10 + i).astype(np.int32)
        return result

    expected_data = [expected_row(i) for i in range(num_rows)]

    expected_data_as_scalars = [{k: v.item() if isinstance(v, np.generic) else v for k, v in row.items()} for row
                                in expected_data]

    # np.datetime64 is converted to a timezone unaware datetime instances. Working explicitly in UTC so we don't need
    # to think about local timezone in the tests
    for row in expected_data_as_scalars:
        row['timestamp'] = row['timestamp'].replace(tzinfo=pytz.UTC)
        if not is_list_of_scalar_broken:
            row['int_fixed_size_list'] = row['int_fixed_size_list'].tolist()
        row['nested_struct'] = {'nested_int': 0}

    rows = [Row(**OrderedDict(sorted(row.items(), key=lambda item: item[0]))) for row in expected_data_as_scalars]

    maybe_int_fixed_size_list_field = [StructField('int_fixed_size_list', ArrayType(IntegerType(), False), False)] \
        if not is_list_of_scalar_broken else []

    # WARNING: surprisingly, schema fields and row fields are matched only by order and not name.
    # We must maintain alphabetical order of the struct fields for the code to work!!!
    schema = StructType(
        [
            StructField('datetime', DateType(), False),
            StructField('float64', DoubleType(), False),
            StructField('id', IntegerType(), False),
            StructField('id_div_700', IntegerType(), False),
        ] + maybe_int_fixed_size_list_field +
        [
            StructField('nested_struct', StructType([StructField('nested_int', IntegerType(), True)])),
            StructField('string', StringType(), False),
            StructField('string2', StringType(), False),
            StructField('timestamp', TimestampType(), False),
        ])

    dataframe = spark.createDataFrame(rows, schema)
    dataframe. \
        coalesce(num_files). \
        write.option('compression', 'none'). \
        mode('overwrite'). \
        partitionBy(*partition_by). \
        parquet(output_url)

    if shutdown:
        spark.stop()

    return expected_data


def create_many_columns_non_petastorm_dataset(output_url, num_rows, num_columns=1000, num_files=4, spark=None):
    """Creates a dataset with the following properties (used in tests)

    1. Has 1000 columns
    2. Each column is an int32 integer
    3. Parquet store consists of 4 files (controlled by ``num_files`` argument)

    :param output_url: The dataset is written to this url (e.g. ``file:///tmp/some_directory``)
    :param num_rows: Number of rows in the generated dataset
    :param num_columns: Number of columns (1000 is the default)
    :param num_files: Number of parquet files that will be created in the store
    :param spark: An instance of SparkSession object. A new instance will be created if non specified
    :return:
    """
    shutdown = False
    if not spark:
        spark_session = SparkSession \
            .builder \
            .appName('petastorm_end_to_end_test') \
            .master('local[*]')

        spark = spark_session.getOrCreate()
        shutdown = True

    column_names = ['col_{}'.format(col_id) for col_id in range(num_columns)]

    def generate_row(i):
        return {'col_{}'.format(col_id): i * 10000 for col_id, col_name in enumerate(column_names)}

    expected_data = [generate_row(row_number) for row_number in range(num_rows)]

    rows = [Row(**row) for row in expected_data]

    # WARNING: surprisingly, schema fields and row fields are matched only by order and not name.
    schema = StructType([StructField(column_name, IntegerType(), False) for column_name in column_names])

    dataframe = spark.createDataFrame(rows, schema)
    dataframe. \
        coalesce(num_files). \
        write.option('compression', 'none'). \
        mode('overwrite'). \
        parquet(output_url)

    if shutdown:
        spark.stop()

    return expected_data
