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
import unittest

import numpy as np
import tensorflow as tf
from pyspark.sql import SparkSession
from pyspark.sql.types import (BinaryType, BooleanType, ByteType, DoubleType,
                               FloatType, IntegerType, LongType, ShortType,
                               StringType, StructField, StructType)
from six.moves.urllib.parse import urlparse

from petastorm.spark.spark_dataset_converter import make_spark_converter


class TfConverterTest(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSession.builder \
            .master("local[2]") \
            .appName("petastorm.spark tests") \
            .getOrCreate()
        self.spark.conf.set("petastorm.spark.converter.defaultCacheDirUrl",
                            "file:///tmp/123")

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
        converter = make_spark_converter(df, 'file:///tmp/123')
        local_path = urlparse(converter.cache_file_path).path
        self.assertTrue(os.path.exists(local_path))
        converter.delete()
        self.assertFalse(os.path.exists(local_path))

    def test_atexit(self):
        cache_dir = "/tmp/spark_converter_test_atexit"
        os.makedirs(cache_dir)
        lines = """
        from petastorm.spark.spark_dataset_converter import make_spark_converter
        from pyspark.sql import SparkSession
        import os
        spark = SparkSession.builder.getOrCreate()
        df = spark.createDataFrame([(1, 2),(4, 5)], ["col1", "col2"])
        converter = make_spark_converter(df, 'file:///tmp/spark_converter_test_atexit')
        f = open("/tmp/spark_converter_test_atexit/output", "w")
        f.write(converter.cache_file_path)
        f.close()
        """
        code_str = "; ".join(
            line.strip() for line in lines.strip().splitlines())
        self.assertTrue(os.path.exists(cache_dir))
        ret_code = subprocess.call(["python", "-c", code_str])
        self.assertEqual(0, ret_code)
        with open(os.path.join(cache_dir, "output")) as f:
            cache_file_path = f.read()
        self.assertFalse(os.path.exists(cache_file_path))

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
                             converter1.cache_file_path).lower())

        converter2 = make_spark_converter(df1, compression=False)
        self.assertEqual("uncompressed",
                         self._get_compression_type(
                             converter2.cache_file_path).lower())

        converter2 = make_spark_converter(df1, compression=True)
        self.assertEqual("snappy",
                         self._get_compression_type(
                             converter2.cache_file_path).lower())

    def test_df_caching(self):
        df1 = self.spark.range(10)
        df2 = self.spark.range(10)
        df3 = self.spark.range(20)

        converter1 = make_spark_converter(df1)
        converter2 = make_spark_converter(df2)
        self.assertEqual(converter1.cache_file_path, converter2.cache_file_path)

        converter3 = make_spark_converter(df3)
        self.assertNotEqual(converter1.cache_file_path,
                            converter3.cache_file_path)

        converter11 = make_spark_converter(
            df1, parquet_row_group_size_bytes=8 * 1024 * 1024)
        converter21 = make_spark_converter(
            df1, parquet_row_group_size_bytes=16 * 1024 * 1024)
        self.assertNotEqual(converter11.cache_file_path,
                            converter21.cache_file_path)

        converter12 = make_spark_converter(df1, compression=True)
        converter22 = make_spark_converter(df1, compression=False)
        self.assertNotEqual(converter12.cache_file_path,
                            converter22.cache_file_path)

    def test_scheme(self):
        url1 = "/tmp/abc"
        url2 = "file:///tmp/123"
        df = self.spark.range(10)

        with self.assertRaises(ValueError) as cm:
            converter = make_spark_converter(df, url1)
            with converter.make_tf_dataset() as _:
                pass
        self.assertEqual("ERROR! A scheme-less dataset url () is no longer "
                         "supported. Please prepend \"file://\" for local "
                         "filesystem.", str(cm.exception))

        converter = make_spark_converter(df, url2)
        with converter.make_tf_dataset() as dataset:
            self.assertIsNotNone(dataset)
