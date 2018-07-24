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

import unittest
from shutil import move, rmtree
from tempfile import mkdtemp

from pyarrow import parquet as pq
from pyspark.sql import SparkSession

from petastorm.etl.dataset_metadata import _generate_num_row_groups_per_file_metadata
from petastorm.fs_utils import FilesystemResolver
from petastorm.reader import Reader
from petastorm.tests.test_common import TestSchema, create_test_dataset
from petastorm.workers_pool.dummy_pool import DummyPool

# Tiny count of rows in a fake dataset
ROWS_COUNT = 10

ORIGINAL_NAME = '_metadata'
NEW_NAME = '_metadata_gone'


def get_test_data_path(subpath, version=0):
    try:
        from av.testutil1 import get_test_data_file
    except ImportError:
        raise unittest.SkipTest('Skipping this test because failed \'from av.testutil1 import get_test_data_file\''
                                'We are probably running from outside of the rAV repo.')

    """ Returns the unit test data path for this unit test. """
    return get_test_data_file('test_metadata_read', version, subpath)


class MetadataUnischemaReadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes dataset once per test. All tests in this class will use the same fake dataset."""
        # Write a fake dataset to this location
        cls._dataset_dir = mkdtemp('test_metadata_read')
        cls._dataset_url = 'file://{}'.format(cls._dataset_dir)
        cls._dataset_dicts = create_test_dataset(cls._dataset_url, range(ROWS_COUNT))

    @classmethod
    def tearDownClass(cls):
        """ Remove everything created in setUpClass. """
        rmtree(cls._dataset_dir)

    def vanish_metadata(self):
        """ Move the already generated _metadata to a different name, leveraging tempdir uniqueness. """
        move('{}/{}'.format(self._dataset_dir, ORIGINAL_NAME), '{}{}'.format(self._dataset_dir, NEW_NAME))

    def restore_metadata(self):
        """ Restore _metadata file for other tests. """
        move('{}{}'.format(self._dataset_dir, NEW_NAME), '{}/{}'.format(self._dataset_dir, ORIGINAL_NAME))

    def test_no_metadata(self):
        self.vanish_metadata()
        with self.assertRaises(ValueError) as e:
            Reader(dataset_url=self._dataset_url, reader_pool=DummyPool())
        self.assertTrue('Could not find _metadata file'in e.exception.message)
        self.restore_metadata()

    def test_metadata_missing_unischema(self):
        """ Produce a BAD _metadata that is missing the unischema pickling first, then load dataset. """
        self.vanish_metadata()
        # We should be able to obtain an existing session
        spark_context = SparkSession.builder.getOrCreate().sparkContext
        # Do all but the last step of petastorm.etl.dataset_metadata.add_dataset_metadata()
        resolver = FilesystemResolver(self._dataset_url)
        dataset = pq.ParquetDataset(
            resolver.parsed_dataset_url().path,
            filesystem=resolver.filesystem(),
            validate_schema=False)
        _generate_num_row_groups_per_file_metadata(dataset, spark_context)
        spark_context.stop()

        with self.assertRaises(ValueError) as e:
            Reader(dataset_url=self._dataset_url, reader_pool=DummyPool())
        self.assertTrue('Could not find the unischema'in e.exception.message)
        self.restore_metadata()

    def test_unischema_loads_from_metadata(self):

        with Reader(dataset_url='file://{}'.format(get_test_data_path('unischema_loads_from_metadata')),
                    reader_pool=DummyPool()) as reader:
            # Check that schema fields are equivalent
            for field in reader.schema.fields:
                self.assertTrue(field in TestSchema.fields)
            for field in TestSchema.fields:
                self.assertTrue(field in reader.schema.fields)


if __name__ == '__main__':
    unittest.main()
