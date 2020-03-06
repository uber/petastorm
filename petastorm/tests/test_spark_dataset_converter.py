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

import os
import subprocess
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from pyspark.sql import SparkSession
from pyspark.sql.types import (BinaryType, BooleanType, ByteType, DoubleType,
                               FloatType, IntegerType, LongType, ShortType,
                               StringType, StructField, StructType)
from six.moves.urllib.parse import urlparse

from petastorm.fs_utils import FilesystemResolver
from petastorm.spark import make_spark_converter
from petastorm.spark.spark_dataset_converter import _check_url, _make_sub_dir_url


class TfConverterTest(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSession.builder \
            .master("local[2]") \
            .appName("petastorm.spark tests") \
            .getOrCreate()
        self.tempdir = tempfile.mkdtemp('_spark_converter_test')
        self.spark.conf.set('petastorm.spark.converter.defaultCacheDirUrl',
                            'file://' + self.tempdir.replace(os.sep, '/'))

    def test_primitive(self):
        schema = StructType([
            StructField("bool_col", BooleanType(), False),
            StructField("float_col", FloatType(), False),
            StructField("double_col", DoubleType(), False),
            StructField("short_col", ShortType(), False),
            StructField("int_col", IntegerType(), False),
            StructField("long_col", LongType(), False),
            StructField("str_col", StringType(), False),
            StructField("bin_col", BinaryType(), False),
            StructField("byte_col", ByteType(), False),
        ])
        df = self.spark.createDataFrame(
            [(True, 0.12, 432.1, 5, 5, 0, "hello",
              bytearray(b"spark\x01\x02"), -128),
             (False, 123.45, 0.987, 9, 908, 765, "petastorm",
              bytearray(b"\x0012345"), 127)],
            schema=schema).coalesce(1)
        # If we use numPartition > 1, the order of the loaded dataset would
        # be non-deterministic.
        expected_df = df.collect()

        converter = make_spark_converter(df)
        with converter.make_tf_dataset() as dataset:
            iterator = dataset.make_one_shot_iterator()
            tensor = iterator.get_next()
            with tf.Session() as sess:
                ts = sess.run(tensor)
                # TODO: we will improve the test once the batch_size argument
                #  added.
                # Now we only have one batch.
            for i in range(converter.dataset_size):
                for col in df.schema.names:
                    actual_ele = getattr(ts, col)[i]
                    expected_ele = expected_df[i][col]
                    if col == "str_col":
                        actual_ele = actual_ele.decode()
                    if col == "bin_col":
                        actual_ele = bytearray(actual_ele)
                    self.assertEqual(expected_ele, actual_ele)

            self.assertEqual(len(expected_df), len(converter))

        self.assertEqual(np.bool_, ts.bool_col.dtype.type,
                         "Boolean type column is not inferred correctly.")
        self.assertEqual(np.float32, ts.float_col.dtype.type,
                         "Float type column is not inferred correctly.")
        self.assertEqual(np.float64, ts.double_col.dtype.type,
                         "Double type column is not inferred correctly.")
        self.assertEqual(np.int16, ts.short_col.dtype.type,
                         "Short type column is not inferred correctly.")
        self.assertEqual(np.int32, ts.int_col.dtype.type,
                         "Integer type column is not inferred correctly.")
        self.assertEqual(np.int64, ts.long_col.dtype.type,
                         "Long type column is not inferred correctly.")
        self.assertEqual(np.object_, ts.str_col.dtype.type,
                         "String type column is not inferred correctly.")
        self.assertEqual(np.object_, ts.bin_col.dtype.type,
                         "Binary type column is not inferred correctly.")

    def test_delete(self):
        df = self.spark.createDataFrame([(1, 2), (4, 5)], ["col1", "col2"])
        # TODO add test for hdfs url
        converter = make_spark_converter(df)
        local_path = urlparse(converter.cache_dir_url).path
        self.assertTrue(os.path.exists(local_path))
        converter.delete()
        self.assertFalse(os.path.exists(local_path))

    def test_atexit(self):
        lines = """
        from petastorm.spark.spark_dataset_converter import make_spark_converter
        from pyspark.sql import SparkSession
        import os
        spark = SparkSession.builder.getOrCreate()
        df = spark.createDataFrame([(1, 2),(4, 5)], ["col1", "col2"])
        converter = make_spark_converter(df)
        f = open(os.join('{tempdir}', 'test_atexit.out'), "w")
        f.write(converter.cache_dir_url)
        f.close()
        """.format(tempdir=self.tempdir)
        code_str = "; ".join(
            line.strip() for line in lines.strip().splitlines())
        ret_code = subprocess.call(["python", "-c", code_str])
        self.assertEqual(0, ret_code)
        with open(os.path.join(self.tempdir, 'test_atexit.out')) as f:
            cache_dir_url = f.read()

        fs = FilesystemResolver(cache_dir_url).filesystem()
        self.assertFalse(fs.exists(urlparse(cache_dir_url).path))

    @staticmethod
    def _get_compression_type(data_url):
        files = os.listdir(urlparse(data_url).path)
        pq_files = list(filter(lambda x: x.endswith('.parquet'), files))
        filename_splits = pq_files[0].split('.')
        if len(filename_splits) == 2:
            return "uncompressed"
        else:
            return filename_splits[1]

    def test_compression(self):
        df1 = self.spark.range(10)

        converter1 = make_spark_converter(df1)
        self.assertEqual("uncompressed",
                         self._get_compression_type(
                             converter1.cache_dir_url).lower())

        converter2 = make_spark_converter(df1, compression_codec="lz4")
        self.assertEqual("lz4",
                         self._get_compression_type(
                             converter2.cache_dir_url).lower())

        converter2 = make_spark_converter(df1, compression_codec="snappy")
        self.assertEqual("snappy",
                         self._get_compression_type(
                             converter2.cache_dir_url).lower())

    def test_df_caching(self):
        df1 = self.spark.range(10)
        df2 = self.spark.range(10)
        df3 = self.spark.range(20)

        converter1 = make_spark_converter(df1)
        converter2 = make_spark_converter(df2)
        self.assertEqual(converter1.cache_dir_url, converter2.cache_dir_url)

        converter3 = make_spark_converter(df3)
        self.assertNotEqual(converter1.cache_dir_url,
                            converter3.cache_dir_url)

        converter11 = make_spark_converter(
            df1, parquet_row_group_size_bytes=8 * 1024 * 1024)
        converter21 = make_spark_converter(
            df1, parquet_row_group_size_bytes=16 * 1024 * 1024)
        self.assertNotEqual(converter11.cache_dir_url,
                            converter21.cache_dir_url)

        converter12 = make_spark_converter(df1, compression_codec=True)
        converter22 = make_spark_converter(df1, compression_codec=False)
        self.assertNotEqual(converter12.cache_dir_url,
                            converter22.cache_dir_url)

    def test_check_url(self):
        with self.assertRaises(ValueError) as cm:
            _check_url('/a/b/c')
        self.assertTrue('scheme-less' in str(cm.exception))

    def test_make_sub_dir_url(self):
        self.assertEquals(_make_sub_dir_url('file:///a/b', 'c'), 'file:///a/b/c')
        self.assertEquals(_make_sub_dir_url('hdfs:/a/b', 'c'), 'hdfs:/a/b/c')
        self.assertEquals(_make_sub_dir_url(
            'hdfs://nn1:9000/a/b', 'c'), 'hdfs://nn1:9000/a/b/c')

    def test_invalid_scheme(self):
        df = self.spark.range(10)

        with self.assertRaises(ValueError) as cm:
            make_spark_converter(df, self.tempdir)

        self.assertTrue('scheme-less' in str(cm.exception))

    def test_pickling_remotely(self):
        df1 = self.spark.range(100, 101)
        converter1 = make_spark_converter(df1)

        def map_fn(_):
            with converter1.make_tf_dataset() as dataset:
                iterator = dataset.make_one_shot_iterator()
                tensor = iterator.get_next()
                with tf.Session() as sess:
                    ts = sess.run(tensor)
            return getattr(ts, 'id')[0]

        result = self.spark.sparkContext.parallelize(range(1), 1).map(map_fn).collect()[0]
        self.assertEqual(result, 100)
