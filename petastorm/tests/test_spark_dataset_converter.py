from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, \
    BooleanType, FloatType, ShortType, IntegerType, LongType, DoubleType

from petastorm.spark.spark_dataset_converter import make_spark_converter

import tensorflow as tf
import numpy as np
import unittest


class TfConverterTest(unittest.TestCase):

    def setUp(self) -> None:
        self.spark = SparkSession.builder \
            .master("local[2]") \
            .appName("petastorm.spark tests") \
            .getOrCreate()

    def test_primitive(self):
        # test primitive columns
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
            (False, 123.45, 0.987, 9, 908, 765)],
            schema=schema)

        converter = make_spark_converter(df)
        with converter.make_tf_dataset() as dataset:
            iterator = dataset.make_one_shot_iterator()
            tensor = iterator.get_next()
            with tf.Session() as sess:
                ts = sess.run(tensor)

        assert (ts.bool_col.dtype.type == np.bool_)
        assert (ts.float_col.dtype.type == np.float32)
        assert (ts.double_col.dtype.type == np.float64)
        assert (ts.short_col.dtype.type == np.int16)
        assert (ts.int_col.dtype.type == np.int32)
        assert (ts.long_col.dtype.type == np.int64)
