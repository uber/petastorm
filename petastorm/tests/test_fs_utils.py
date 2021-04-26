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

import dill
from pyarrow.filesystem import LocalFileSystem
from pyarrow.lib import ArrowIOError
from six.moves.urllib.parse import urlparse
import gcsfs
import s3fs

from petastorm.fs_utils import FilesystemResolver, get_filesystem_and_path_or_paths
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
        cls.mock_name = "mock-manager"

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
            FilesystemResolver(urlparse('unknown://foo/bar'), {})

        # s3 paths must have the bucket as the netloc
        with self.assertRaises(ValueError):
            FilesystemResolver(urlparse('s3:///foo/bar'), {})

        # GCS paths must have the bucket as the netloc
        with self.assertRaises(ValueError):
            FilesystemResolver(urlparse('gcs:///foo/bar'), {})

    def test_file_url(self):
        """ Case 2: File path, agnostic to content of hadoop configuration."""
        suj = FilesystemResolver('file://{}'.format(ABS_PATH), self._hadoop_configuration, connector=self.mock)
        self.assertTrue(isinstance(suj.filesystem(), LocalFileSystem))
        self.assertEqual('', suj.parsed_dataset_url().netloc)
        self.assertEqual(ABS_PATH, suj.get_dataset_path())

        # Make sure we did not capture FilesystemResolver in a closure by mistake
        dill.dumps(suj.filesystem_factory())

    def test_hdfs_url_with_nameservice(self):
        """ Case 3a: HDFS nameservice."""
        suj = FilesystemResolver(HC.WARP_TURTLE_PATH, self._hadoop_configuration, connector=self.mock,
                                 user=self.mock_name)
        self.assertEqual(MockHdfs, type(suj.filesystem()._hdfs))
        self.assertEqual(self.mock_name, suj.filesystem()._user)
        self.assertEqual(HC.WARP_TURTLE, suj.parsed_dataset_url().netloc)
        self.assertEqual(1, self.mock.connect_attempted(HC.WARP_TURTLE_NN2))
        self.assertEqual(0, self.mock.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(0, self.mock.connect_attempted(HC.DEFAULT_NN))

        # Make sure we did not capture FilesystemResolver in a closure by mistake
        dill.dumps(suj.filesystem_factory())

    def test_hdfs_url_no_nameservice(self):
        """ Case 3b: HDFS with no nameservice should connect to default namenode."""
        suj = FilesystemResolver('hdfs:///some/path', self._hadoop_configuration, connector=self.mock,
                                 user=self.mock_name)
        self.assertEqual(MockHdfs, type(suj.filesystem()._hdfs))
        self.assertEqual(self.mock_name, suj.filesystem()._user)
        self.assertEqual(HC.WARP_TURTLE, suj.parsed_dataset_url().netloc)
        # ensure path is preserved in parsed URL
        self.assertEqual('/some/path', suj.get_dataset_path())
        self.assertEqual(1, self.mock.connect_attempted(HC.WARP_TURTLE_NN2))
        self.assertEqual(0, self.mock.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(0, self.mock.connect_attempted(HC.DEFAULT_NN))

        # Make sure we did not capture FilesystemResolver in a closure by mistake
        dill.dumps(suj.filesystem_factory())

    def test_hdfs_url_direct_namenode(self):
        """ Case 4: direct namenode."""
        suj = FilesystemResolver('hdfs://{}/path'.format(HC.WARP_TURTLE_NN1),
                                 self._hadoop_configuration,
                                 connector=self.mock,
                                 user=self.mock_name)
        self.assertEqual(MockHdfs, type(suj.filesystem()))
        self.assertEqual(self.mock_name, suj.filesystem()._user)
        self.assertEqual(HC.WARP_TURTLE_NN1, suj.parsed_dataset_url().netloc)
        self.assertEqual(0, self.mock.connect_attempted(HC.WARP_TURTLE_NN2))
        self.assertEqual(1, self.mock.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(0, self.mock.connect_attempted(HC.DEFAULT_NN))

        # Make sure we did not capture FilesystemResolver in a closure by mistake
        dill.dumps(suj.filesystem_factory())

    def test_hdfs_url_direct_namenode_driver_libhdfs(self):
        suj = FilesystemResolver('hdfs://{}/path'.format(HC.WARP_TURTLE_NN1),
                                 self._hadoop_configuration,
                                 connector=self.mock, hdfs_driver='libhdfs', user=self.mock_name)
        self.assertEqual(MockHdfs, type(suj.filesystem()))
        self.assertEqual(self.mock_name, suj.filesystem()._user)
        # Make sure we did not capture FilesystemResolver in a closure by mistake
        dill.dumps(suj.filesystem_factory())

    def test_hdfs_url_direct_namenode_retries(self):
        """ Case 4: direct namenode fails first two times thru, but 2nd retry succeeds."""
        self.mock.set_fail_n_next_connect(2)
        with self.assertRaises(ArrowIOError):
            suj = FilesystemResolver('hdfs://{}/path'.format(HC.WARP_TURTLE_NN2),
                                     self._hadoop_configuration,
                                     connector=self.mock, user=self.mock_name)
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
                                 connector=self.mock, user=self.mock_name)
        self.assertEqual(MockHdfs, type(suj.filesystem()))
        self.assertEqual(self.mock_name, suj.filesystem()._user)
        self.assertEqual(HC.WARP_TURTLE_NN2, suj.parsed_dataset_url().netloc)
        self.assertEqual(3, self.mock.connect_attempted(HC.WARP_TURTLE_NN2))
        self.assertEqual(0, self.mock.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(0, self.mock.connect_attempted(HC.DEFAULT_NN))

    def test_s3_url(self):
        suj = FilesystemResolver('s3://bucket{}'.format(ABS_PATH), self._hadoop_configuration, connector=self.mock)
        self.assertTrue(isinstance(suj.filesystem(), s3fs.S3FileSystem))
        self.assertEqual('bucket', suj.parsed_dataset_url().netloc)
        self.assertEqual('bucket' + ABS_PATH, suj.get_dataset_path())

        # Make sure we did not capture FilesystemResolver in a closure by mistake
        dill.dumps(suj.filesystem_factory())

    def test_gcs_url(self):
        suj = FilesystemResolver('gcs://bucket{}'.format(ABS_PATH), self._hadoop_configuration, connector=self.mock)
        self.assertTrue(isinstance(suj.filesystem(), gcsfs.GCSFileSystem))
        self.assertEqual('bucket', suj.parsed_dataset_url().netloc)
        self.assertEqual('bucket' + ABS_PATH, suj.get_dataset_path())

        # Make sure we did not capture FilesystemResolver in a closure by mistake
        dill.dumps(suj.filesystem_factory())

    def test_get_filesystem_and_path_or_paths(self):
        fs1, path1 = get_filesystem_and_path_or_paths('file:///some/path')
        assert isinstance(fs1, LocalFileSystem) and path1 == '/some/path'

        fs2, paths2 = get_filesystem_and_path_or_paths(
            ['file:///some/path/01.parquet', 'file:///some/path/02.parquet'])
        assert isinstance(fs2, LocalFileSystem) and \
            paths2 == ['/some/path/01.parquet', '/some/path/02.parquet']

        with self.assertRaises(ValueError):
            get_filesystem_and_path_or_paths(
                ['file:///some/path/01.parquet', 'hdfs:///some/path/02.parquet'])


if __name__ == '__main__':
    unittest.main()
