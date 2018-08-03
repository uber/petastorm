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

from pyarrow.filesystem import LocalFileSystem
from pyarrow.lib import ArrowIOError
from six.moves.urllib.parse import urlparse

from petastorm.fs_utils import FilesystemResolver
from petastorm.hdfs.tests.test_hdfs_namenode import HC, MockHadoopConfiguration, \
    MockHdfs, MockHdfsConnector

ABS_PATH = '/abs/path'


class FilesystemResolverTest(unittest.TestCase):
    """
    Checks the full filesystem resolution functionality, exercising each URL interpretation case.
    """

    @classmethod
    def setUpClass(cls):
        cls.mock = MockHdfsConnector()

    def setUp(self):
        """Initializes a mock hadoop config and populate with basic properties."""
        # Reset counters in mock connector
        self.mock.reset()
        self._hadoop_configuration = MockHadoopConfiguration()
        self._hadoop_configuration.set('fs.defaultFS', HC.FS_WARP_TURTLE)
        self._hadoop_configuration.set('dfs.ha.namenodes.{}'.format(HC.WARP_TURTLE), 'nn2,nn1')
        self._hadoop_configuration.set('dfs.namenode.rpc-address.{}.nn1'.format(HC.WARP_TURTLE), HC.WARP_TURTLE_NN1)
        self._hadoop_configuration.set('dfs.namenode.rpc-address.{}.nn2'.format(HC.WARP_TURTLE), HC.WARP_TURTLE_NN2)

    def test_error_url_cases(self):
        """Various error cases that result in exception raised."""
        # Case 1: Schemeless path asserts
        with self.assertRaises(ValueError):
            FilesystemResolver(ABS_PATH, {})

        # Case 4b: HDFS default path case with NO defaultFS
        with self.assertRaises(RuntimeError):
            FilesystemResolver('hdfs:///some/path', {})

        # Case 4b: Using `default` as host, while apparently a pyarrow convention, is NOT valid
        with self.assertRaises(ArrowIOError):
            FilesystemResolver('hdfs://default', {})

        # Case 5: other schemes result in ValueError; urlparse to cover an else branch!
        with self.assertRaises(ValueError):
            FilesystemResolver(urlparse('http://foo/bar'), {})
        with self.assertRaises(ValueError):
            FilesystemResolver(urlparse('ftp://foo/bar'), {})
        with self.assertRaises(ValueError):
            FilesystemResolver(urlparse('s3://foo/bar'), {})
        with self.assertRaises(ValueError):
            FilesystemResolver(urlparse('ssh://foo/bar'), {})

    def test_file_url(self):
        """ Case 2: File path, agnostic to content of hadoop configuration."""
        suj = FilesystemResolver('file://{}'.format(ABS_PATH), self._hadoop_configuration, connector=self.mock)
        self.assertTrue(isinstance(suj.filesystem(), LocalFileSystem))
        self.assertEqual('', suj.parsed_dataset_url().netloc)
        self.assertEqual(ABS_PATH, suj.parsed_dataset_url().path)

    def test_hdfs_url_with_nameservice(self):
        """ Case 3a: HDFS nameservice."""
        suj = FilesystemResolver(HC.WARP_TURTLE_PATH, self._hadoop_configuration, connector=self.mock)
        self.assertEqual(MockHdfs, type(suj.filesystem()._hdfs))
        self.assertEqual(HC.WARP_TURTLE, suj.parsed_dataset_url().netloc)
        self.assertEqual(1, self.mock.connect_attempted(HC.WARP_TURTLE_NN2))
        self.assertEqual(0, self.mock.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(0, self.mock.connect_attempted(HC.DEFAULT_NN))

    def test_hdfs_url_no_nameservice(self):
        """ Case 3b: HDFS with no nameservice should connect to default namenode."""
        suj = FilesystemResolver('hdfs:///some/path', self._hadoop_configuration, connector=self.mock)
        self.assertEqual(MockHdfs, type(suj.filesystem()._hdfs))
        self.assertEqual(HC.WARP_TURTLE, suj.parsed_dataset_url().netloc)
        # ensure path is preserved in parsed URL
        self.assertEqual('/some/path', suj.parsed_dataset_url().path)
        self.assertEqual(1, self.mock.connect_attempted(HC.WARP_TURTLE_NN2))
        self.assertEqual(0, self.mock.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(0, self.mock.connect_attempted(HC.DEFAULT_NN))

    def test_hdfs_url_direct_namenode(self):
        """ Case 4: direct namenode."""
        suj = FilesystemResolver('hdfs://{}/path'.format(HC.WARP_TURTLE_NN1),
                                 self._hadoop_configuration,
                                 connector=self.mock)
        self.assertEqual(MockHdfs, type(suj.filesystem()))
        self.assertEqual(HC.WARP_TURTLE_NN1, suj.parsed_dataset_url().netloc)
        self.assertEqual(0, self.mock.connect_attempted(HC.WARP_TURTLE_NN2))
        self.assertEqual(1, self.mock.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(0, self.mock.connect_attempted(HC.DEFAULT_NN))

    def test_hdfs_url_direct_namenode_retries(self):
        """ Case 4: direct namenode fails first two times thru, but 2nd retry succeeds."""
        self.mock.set_fail_n_next_connect(2)
        with self.assertRaises(ArrowIOError):
            suj = FilesystemResolver('hdfs://{}/path'.format(HC.WARP_TURTLE_NN2),
                                     self._hadoop_configuration,
                                     connector=self.mock)
        self.assertEqual(1, self.mock.connect_attempted(HC.WARP_TURTLE_NN2))
        self.assertEqual(0, self.mock.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(0, self.mock.connect_attempted(HC.DEFAULT_NN))
        with self.assertRaises(ArrowIOError):
            suj = FilesystemResolver('hdfs://{}/path'.format(HC.WARP_TURTLE_NN2),
                                     self._hadoop_configuration,
                                     connector=self.mock)
        self.assertEqual(2, self.mock.connect_attempted(HC.WARP_TURTLE_NN2))
        self.assertEqual(0, self.mock.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(0, self.mock.connect_attempted(HC.DEFAULT_NN))
        # this one should connect "successfully"
        suj = FilesystemResolver('hdfs://{}/path'.format(HC.WARP_TURTLE_NN2),
                                 self._hadoop_configuration,
                                 connector=self.mock)
        self.assertEqual(MockHdfs, type(suj.filesystem()))
        self.assertEqual(HC.WARP_TURTLE_NN2, suj.parsed_dataset_url().netloc)
        self.assertEqual(3, self.mock.connect_attempted(HC.WARP_TURTLE_NN2))
        self.assertEqual(0, self.mock.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(0, self.mock.connect_attempted(HC.DEFAULT_NN))


if __name__ == '__main__':
    unittest.main()
