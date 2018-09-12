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
import os
import unittest
from shutil import move, rmtree
from tempfile import mkdtemp

from petastorm.etl.dataset_metadata import PetastormMetadataError
from petastorm.reader import Reader
from petastorm.tests.test_common import create_test_dataset
from petastorm.workers_pool.dummy_pool import DummyPool

# Tiny count of rows in a fake dataset
ROWS_COUNT = 10


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

    def vanish_metadata(self, filename='_common_metadata'):
        """ Move the already generated _metadata to a different name, leveraging tempdir uniqueness. """
        move('{}/{}'.format(self._dataset_dir, filename), '{}{}'.format(self._dataset_dir, filename + '_gone'))

    def restore_metadata(self, filename='_common_metadata'):
        """ Restore _metadata file for other tests. """
        move('{}{}'.format(self._dataset_dir, filename + '_gone'), '{}/{}'.format(self._dataset_dir, filename))

    def test_no_common_metadata_crc(self):
        """We add our own entries to the _common_metadata file, unfortunatelly, the .crc file is not updated by
        current pyarrow implementation, so we delete the .crc to make sure there is no mismatch with the content of
        _common_metadata file"""
        self.assertFalse(os.path.exists(os.path.join(MetadataUnischemaReadTest._dataset_dir, '._common_metadata.crc')))

    def test_no_metadata(self):
        self.vanish_metadata()
        with self.assertRaises(PetastormMetadataError) as e:
            Reader(self._dataset_url, reader_pool=DummyPool())
        self.assertTrue('Could not find _common_metadata file' in str(e.exception))
        self.restore_metadata()


if __name__ == '__main__':
    unittest.main()
