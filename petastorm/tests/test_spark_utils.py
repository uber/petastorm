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
import glob
import os
import unittest
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
from pyspark.sql import SparkSession

from petastorm.spark_utils import dataset_as_rdd
from petastorm.tests.test_common import create_test_dataset, TestSchema


class TestSparkUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initializes dataset once per test. All tests in this class will use the same fake dataset."""
        # Write a fake dataset to this location
        cls._dataset_dir = mkdtemp('end_to_end_petastorm')
        cls._dataset_url = 'file://{}'.format(cls._dataset_dir)
        ROWS_COUNT = 1000
        cls._dataset_dicts = create_test_dataset(cls._dataset_url, range(ROWS_COUNT))

        # Remove crc files due to https://issues.apache.org/jira/browse/HADOOP-7199
        for crc_file in glob.glob(cls._dataset_dir + '/.*.crc'):
            os.remove(crc_file)

    @classmethod
    def tearDownClass(cls):
        # Remove everything created with "get_temp_dir"
        rmtree(cls._dataset_dir)

    def _get_spark_session(self):
        return SparkSession \
            .builder \
            .appName('petastorm_spark_utils_test') \
            .master('local[*]')\
            .getOrCreate()

    def test_simple_read_rdd(self):
        """Read dataset into spark rdd. Collects and makes sure they all return as expected"""
        spark = self._get_spark_session()
        rows = dataset_as_rdd(self._dataset_url, spark).collect()

        for row in rows:
            actual = dict(row._asdict())
            expected = next(d for d in self._dataset_dicts if d['id'] == actual['id'])
            np.testing.assert_equal(expected, actual)

        spark.stop()

    def test_reading_subset_of_columns(self):
        """Read subset of dataset fields into spark rdd. Collects and makes sure they all return as expected"""
        spark = self._get_spark_session()
        rows = dataset_as_rdd(self._dataset_url, spark, schema_fields=[TestSchema.id2, TestSchema.id]).collect()

        for row in rows:
            actual = dict(row._asdict())
            expected = next(d for d in self._dataset_dicts if d['id'] == actual['id'])
            np.testing.assert_equal(expected['id2'], actual['id2'])

        spark.stop()


if __name__ == '__main__':
    # Delegate to the test framework.
    unittest.main()
