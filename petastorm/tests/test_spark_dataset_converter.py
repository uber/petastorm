from petastorm.spark.spark_dataset_converter import make_spark_converter, SparkDatasetConverter
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, \
    BooleanType, FloatType, ShortType, IntegerType, LongType, DoubleType

import numpy as np
import os
import tensorflow as tf
import unittest


class TfConverterTest(unittest.TestCase):

    def setUp(self) -> None:
        self.spark = SparkSession.builder \
            .master("local[2]") \
            .appName("petastorm.spark tests") \
            .getOrCreate()

    def test_primitive(self):
        schema = StructType([
            StructField("bool_col", BooleanType(), False),
            StructField("float_col", FloatType(), False),
            StructField("double_col", DoubleType(), False),
            StructField("short_col", ShortType(), False),
            StructField("int_col", IntegerType(), False),
            StructField("long_col", LongType(), False)
        ])
        df = self.spark.createDataFrame([
            (True, 0.12, 432.1, 5, 5, 0),
            (False, 123.45, 0.987, 9, 908, 765)], schema=schema).coalesce(1)
        # If we use numPartition > 1, the order of the loaded dataset would be non-deterministic.
        expected_df = df.collect()

        converter = make_spark_converter(df)
        with converter.make_tf_dataset() as dataset:
            iterator = dataset.make_one_shot_iterator()
            tensor = iterator.get_next()
            with tf.Session() as sess:
                ts = sess.run(tensor)
                # TODO: we will improve the test once the batch_size argument added.
                # Now we only have one batch.
            for i in range(converter.dataset_size):
                for col in df.schema.names:
                    self.assertEquals(getattr(ts, col)[i], expected_df[i][col])

            self.assertEquals(len(converter), len(expected_df))

        self.assertEquals(ts.bool_col.dtype.type, np.bool_, "Boolean type column is not inferred correctly.")
        self.assertEquals(ts.float_col.dtype.type, np.float32, "Float type column is not inferred correctly.")
        self.assertEquals(ts.double_col.dtype.type, np.float64, "Double type column is not inferred correctly.")
        self.assertEquals(ts.short_col.dtype.type, np.int16, "Short type column is not inferred correctly.")
        self.assertEquals(ts.int_col.dtype.type, np.int32, "Integer type column is not inferred correctly.")
        self.assertEquals(ts.long_col.dtype.type, np.int64, "Long type column is not inferred correctly.")

    def test_delete(self):
        test_path = "/tmp/petastorm_test"
        os.mkdir(test_path)
        os.mkdir(os.path.join(test_path, "dir1"))
        with open(os.path.join(test_path, "file1"), "w") as f:
            f.write("abc")
        with open(os.path.join(test_path, "file2"), "w") as f:
            f.write("123")
        converter = SparkDatasetConverter(test_path, 0)
        converter.delete()
        self.assertFalse(os.path.exists(test_path))
