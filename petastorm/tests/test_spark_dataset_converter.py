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
from pyspark.sql.types import (ArrayType, BinaryType, BooleanType, ByteType,
                               DoubleType, FloatType, IntegerType, LongType,
                               ShortType, StringType, StructField, StructType)
from six.moves.urllib.parse import urlparse

from petastorm import make_spark_converter
from petastorm.spark.spark_dataset_converter import _normalize_dir_url, _is_sub_dir_url


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

                    if col == "float_col" or col == "double_col":
                        # Note that the default precision is float32
                        self.assertAlmostEqual(expected_ele, actual_ele, delta=1e-5)
                    else:
                        self.assertEqual(expected_ele, actual_ele)

            self.assertEqual(len(expected_df), len(converter))

        self.assertEqual(np.bool_, ts.bool_col.dtype.type,
                         "Boolean type column is not inferred correctly.")
        self.assertEqual(np.float32, ts.float_col.dtype.type,
                         "Float type column is not inferred correctly.")
        # Default precision float32
        self.assertEqual(np.float32, ts.double_col.dtype.type,
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
        local_path = urlparse(converter.cache_dir_url).path
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
        f.write(converter.cache_dir_url)
        f.close()
        """
        code_str = "; ".join(
            line.strip() for line in lines.strip().splitlines())
        self.assertTrue(os.path.exists(cache_dir))
        ret_code = subprocess.call(["python", "-c", code_str])
        self.assertEqual(0, ret_code)
        with open(os.path.join(cache_dir, "output")) as f:
            cache_dir_url = f.read()
        self.assertFalse(os.path.exists(cache_dir_url))

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

        converter2 = make_spark_converter(df1, compression=False)
        self.assertEqual("uncompressed",
                         self._get_compression_type(
                             converter2.cache_dir_url).lower())

        converter2 = make_spark_converter(df1, compression=True)
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

        converter12 = make_spark_converter(df1, compression=True)
        converter22 = make_spark_converter(df1, compression=False)
        self.assertNotEqual(converter12.cache_dir_url,
                            converter22.cache_dir_url)

    def test_normalize_url(self):
        with self.assertRaises(ValueError) as cm:
            _normalize_dir_url('/a/b/c')
        self.assertTrue('scheme-less' in str(cm.exception))

        self.assertEqual(_normalize_dir_url('file:///a/b'), 'file:///a/b')
        self.assertEqual(_normalize_dir_url('file:///a//b'), 'file:///a/b')

    def test_df_private_caching(self):
        self.assertTrue(_is_sub_dir_url('file:///a/b/c/d', 'file:///a/b/c'))
        self.assertTrue(_is_sub_dir_url('hdfs:///a/b/c/d', 'hdfs:///a/b/c'))
        self.assertTrue(_is_sub_dir_url('hdfs://nn1:9000/a/b/c/d', 'hdfs://nn1:9000/a/b/c'))

        self.assertFalse(_is_sub_dir_url('file:///a/b/c', 'file:///a/b/c'))
        self.assertFalse(_is_sub_dir_url('file:///a/b/c/d', 'file:///a/b/cc'))
        self.assertFalse(_is_sub_dir_url('file:///a/b/c', 'file:///a/b/c/d'))
        self.assertFalse(_is_sub_dir_url('file:///a/b/c', 'hdfs:///a/b/c'))
        self.assertFalse(_is_sub_dir_url('hdfs://nn1:9000/a/b/c/d', 'hdfs://nn1:9001/a/b/c'))

        df1 = self.spark.range(10)
        df2 = self.spark.range(10)

        converter1 = make_spark_converter(df1, cache_dir_url='file:///tmp/a1/a2')
        converter2 = make_spark_converter(df2, cache_dir_url='file:///tmp/a1/a2')
        converter3 = make_spark_converter(df2, cache_dir_url='file:///tmp/a1/')
        converter4 = make_spark_converter(df2, cache_dir_url='file:///tmp/a1/a3')

        self.assertEqual(converter1.cache_dir_url, converter2.cache_dir_url)
        self.assertEqual(converter1.cache_dir_url, converter3.cache_dir_url)
        self.assertNotEqual(converter1.cache_dir_url, converter4.cache_dir_url)

    def test_scheme(self):
        url1 = "/tmp/abc"
        url2 = "file:///tmp/123"
        df = self.spark.range(10)

        with self.assertRaises(ValueError) as cm:
            converter = make_spark_converter(df, url1)
            with converter.make_tf_dataset() as _:
                pass
        self.assertTrue('scheme-less' in str(cm.exception))

        converter = make_spark_converter(df, url2)
        with converter.make_tf_dataset() as dataset:
            self.assertIsNotNone(dataset)

    def test_pickling_remotely(self):
        df1 = self.spark.range(100, 101)
        converter1 = make_spark_converter(df1)

        def map_fn(_):
            with converter1.make_tf_dataset() as dataset:
                iterator = dataset.make_one_shot_iterator()
                tensor = iterator.get_next()
                with tf.Session() as sess:
                    ts = sess.run(tensor)
            return ts.id[0]

        result = self.spark.sparkContext.parallelize(range(1), 1).map(map_fn).collect()[0]
        self.assertEqual(result, 100)

    def test_tf_dataset_batch_size(self):
        df1 = self.spark.range(100)

        batch_size = 30
        converter1 = make_spark_converter(df1)

        with converter1.make_tf_dataset(batch_size) as dataset:
            iterator = dataset.make_one_shot_iterator()
            tensor = iterator.get_next()
            with tf.Session() as sess:
                ts = sess.run(tensor)
        self.assertEqual(len(ts.id), batch_size)

    def test_tf_dataset_preproc(self):
        df1 = self.spark.createDataFrame(
            [([1., 2., 3., 4., 5., 6.],),
             ([4., 5., 6., 7., 8., 9.],)],
            StructType([StructField(name='c1', dataType=ArrayType(DoubleType()))]))

        converter1 = make_spark_converter(df1)

        def preproc_fn(x):
            return tf.reshape(x.c1, [-1, 3, 2]),

        with converter1.make_tf_dataset(batch_size=2, preproc_fn=preproc_fn) as dataset:
            iterator = dataset.make_one_shot_iterator()
            tensor = iterator.get_next()
            with tf.Session() as sess:
                ts = sess.run(tensor)

        self.assertEqual(ts[0].shape, (2, 3, 2))

    def test_precision(self):
        df = self.spark.range(10)
        df = df.withColumn("float_col", df.id.cast(FloatType())) \
            .withColumn("double_col", df.id.cast(DoubleType()))

        converter1 = make_spark_converter(df)
        with converter1.make_tf_dataset() as dataset:
            iterator = dataset.make_one_shot_iterator()
            tensor = iterator.get_next()
            with tf.Session() as sess:
                ts = sess.run(tensor)
        self.assertEqual(np.float32, ts.double_col.dtype.type)

        converter2 = make_spark_converter(df, precision="float64")
        with converter2.make_tf_dataset() as dataset:
            iterator = dataset.make_one_shot_iterator()
            tensor = iterator.get_next()
            with tf.Session() as sess:
                ts = sess.run(tensor)
        self.assertEqual(np.float64, ts.float_col.dtype.type)

        with self.assertRaises(ValueError) as cm:
            make_spark_converter(df, precision="float16")
            self.assertIn("precision float16 is not supported. \
                Use 'float32' or float64", str(cm.exception))

    def test_array(self):
        df = self.spark.createDataFrame(
            [([1., 2., 3.],)],
            StructType([
                StructField(name='c1', dataType=ArrayType(DoubleType()))
            ])
        )
        converter1 = make_spark_converter(df)
        with converter1.make_tf_dataset() as dataset:
            iterator = dataset.make_one_shot_iterator()
            tensor = iterator.get_next()
            with tf.Session() as sess:
                ts = sess.run(tensor)
        self.assertEqual(np.float32, ts.c1.dtype.type)
