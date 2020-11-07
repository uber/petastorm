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

from pyarrow import fs

from petastorm.fs_utils import get_filesystem_and_path_or_paths
from petastorm.hdfs.tests.test_hdfs_namenode import HC, MockHadoopConfiguration, \
    MockHdfsConnector

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

    def test_get_filesystem_and_path_or_paths(self):
        fs1, path1 = get_filesystem_and_path_or_paths('file:///some/path')
        assert isinstance(fs1, fs.LocalFileSystem) and path1 == '/some/path'

        fs2, paths2 = get_filesystem_and_path_or_paths(['file:///some/path/01.parquet', 'file:///some/path/02.parquet'])
        assert isinstance(fs2, fs.LocalFileSystem) and paths2 == ['/some/path/01.parquet', '/some/path/02.parquet']

        with self.assertRaises(ValueError):
            get_filesystem_and_path_or_paths(
                ['file:///some/path/01.parquet', 'hdfs:///some/path/02.parquet'])


if __name__ == '__main__':
    unittest.main()
