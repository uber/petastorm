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
import pickle
import textwrap
import unittest

import pytest
from pyarrow.lib import ArrowIOError

try:
    from unittest import mock
except ImportError:
    from mock import mock

from petastorm.hdfs.namenode import HdfsNamenodeResolver, HdfsConnector, \
    HdfsConnectError, MaxFailoversExceeded, HAHdfsClient, namenode_failover


class HC:
    """Hadoop constants for testing convenience"""
    WARP_TURTLE = 'WARP-TURTLE'
    FS_WARP_TURTLE = 'hdfs://{}'.format(WARP_TURTLE)
    DEFAULT_NN = 'default:8020'
    WARP_TURTLE_NN1 = 'some.domain.name.net:8020'
    WARP_TURTLE_NN2 = 'some.other.domain.name.net:8020'
    WARP_TURTLE_PATH = '{}/x/y/z'.format(FS_WARP_TURTLE)
    HADOOP_CONFIG_PATH = '/etc/hadoop'


class MockHadoopConfiguration(object):
    def __init__(self):
        self._dict = {}

    def get(self, key):
        val = None
        if key in self._dict:
            val = self._dict[key]
        # print('MockHadoopConfiguration: "{}" == "{}"'.format(key, val))
        return val

    def set(self, key, val):
        self._dict[key] = val


class HdfsNamenodeResolverTest(unittest.TestCase):
    def setUp(self):
        """Initializes a mock hadoop config and a namenode resolver instance, for convenience."""
        self._hadoop_configuration = MockHadoopConfiguration()
        self.suj = HdfsNamenodeResolver(self._hadoop_configuration)

    def test_default_hdfs_service_errors(self):
        """Check error cases with connecting to default namenode"""
        # No default yields RuntimeError
        with self.assertRaises(RuntimeError):
            self.suj.resolve_default_hdfs_service()
        # Bad default FS yields IOError
        self._hadoop_configuration.set('fs.defaultFS', 'invalidFS')
        with self.assertRaises(IOError):
            self.suj.resolve_default_hdfs_service()
        # Random FS host yields IOError
        self._hadoop_configuration.set('fs.defaultFS', 'hdfs://random')
        with self.assertRaises(IOError):
            self.suj.resolve_default_hdfs_service()
        # Valid FS host with no namenode defined yields IOError
        self._hadoop_configuration.set('fs.defaultFS', HC.FS_WARP_TURTLE)
        with self.assertRaises(IOError):
            self.suj.resolve_default_hdfs_service()

    def test_default_hdfs_service_typical(self):
        """Check typical cases resolving default namenode"""
        # One nn
        self._hadoop_configuration.set('fs.defaultFS', HC.FS_WARP_TURTLE)
        self._hadoop_configuration.set('dfs.ha.namenodes.{}'.format(HC.WARP_TURTLE), 'nn1')
        self._hadoop_configuration.set(
            'dfs.namenode.rpc-address.{}.nn1'.format(HC.WARP_TURTLE), HC.WARP_TURTLE_NN1)
        nameservice, namenodes = self.suj.resolve_default_hdfs_service()
        self.assertEqual(HC.WARP_TURTLE, nameservice)
        self.assertEqual(HC.WARP_TURTLE_NN1, namenodes[0])

        # Second of two nns, when the first is undefined
        self._hadoop_configuration.set('dfs.ha.namenodes.{}'.format(HC.WARP_TURTLE), 'nn2,nn1')
        with self.assertRaises(RuntimeError):
            self.suj.resolve_default_hdfs_service()

        # Two valid and defined nns
        self._hadoop_configuration.set(
            'dfs.namenode.rpc-address.{}.nn2'.format(HC.WARP_TURTLE), HC.WARP_TURTLE_NN2)
        nameservice, namenodes = self.suj.resolve_default_hdfs_service()
        self.assertEqual(HC.WARP_TURTLE, nameservice)
        self.assertEqual(HC.WARP_TURTLE_NN2, namenodes[0])
        self.assertEqual(HC.WARP_TURTLE_NN1, namenodes[1])

    def test_resolve_hdfs_name_service(self):
        """Check edge cases with resolving a nameservice"""
        # Most cases already covered by test_default_hdfs_service_ok above...
        # Empty config or no namespace yields None
        self.assertIsNone(HdfsNamenodeResolver({}).resolve_hdfs_name_service(''))
        self.assertIsNone(self.suj.resolve_hdfs_name_service(''))

        # Test a single undefined namenode case, as well as an unconventional multi-NN case;
        # both result in an exception raised
        self._hadoop_configuration.set('fs.defaultFS', HC.FS_WARP_TURTLE)
        self._hadoop_configuration.set('dfs.ha.namenodes.{}'.format(HC.WARP_TURTLE), 'nn1')
        with self.assertRaises(RuntimeError):
            self.suj.resolve_hdfs_name_service(HC.WARP_TURTLE)

        # Test multiple undefined NNs, which will also throw HdfsConnectError
        nns = 'nn1,nn2,nn3,nn4,nn5,nn6,nn7,nn8'
        self._hadoop_configuration.set('dfs.ha.namenodes.{}'.format(HC.WARP_TURTLE), nns)
        with self.assertRaises(RuntimeError):
            self.suj.resolve_hdfs_name_service(HC.WARP_TURTLE)


@pytest.fixture()
def mock_hadoop_home_directory(tmpdir):
    """Create hadoop site files once"""
    tmpdir_path = tmpdir.strpath
    os.makedirs('{}{}'.format(tmpdir_path, HC.HADOOP_CONFIG_PATH))
    with open('{}{}/core-site.xml'.format(tmpdir_path, HC.HADOOP_CONFIG_PATH), 'wt') as f:
        f.write(textwrap.dedent("""\
            <?xml version="1.0"?>
            <?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
            <configuration>
              <property>
                <name>fs.defaultFS</name>
                <value>hdfs://{0}</value>
              </property>
            </configuration>
            """.format(HC.WARP_TURTLE)))
    with open('{}{}/hdfs-site.xml'.format(tmpdir_path, HC.HADOOP_CONFIG_PATH), 'wt') as f:
        f.write(textwrap.dedent("""\
            <?xml version="1.0"?>
            <?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
            <configuration>
              <property>
                <name>dfs.ha.namenodes.{0}</name>
                <value>nn2,nn1</value>
              </property>
              <property>
                <name>dfs.namenode.rpc-address.{0}.nn1</name>
                <value>{1}</value>
              </property>
              <property>
                <name>dfs.namenode.rpc-address.{0}.nn2</name>
                <value>{2}</value>
              </property>
              <property>
                <name>dfs.ha.namenodes.foobar</name>
                <value>nn</value>
              </property>
            </configuration>
            """.format(HC.WARP_TURTLE, HC.WARP_TURTLE_NN1, HC.WARP_TURTLE_NN2)))
    return tmpdir_path


def _test_default_hdfs_service(mock_hadoop_home_directory, env_var):
    # Trigger env var evaluation
    suj = HdfsNamenodeResolver()
    assert env_var == suj._hadoop_env
    assert mock_hadoop_home_directory == suj._hadoop_path
    # List of namenodes returned nominally
    nameservice, namenodes = suj.resolve_default_hdfs_service()
    assert HC.WARP_TURTLE == nameservice
    assert HC.WARP_TURTLE_NN2 == namenodes[0]
    assert HC.WARP_TURTLE_NN1 == namenodes[1]
    # Exception raised for badly defined nameservice (XML issue)
    with pytest.raises(RuntimeError):
        suj.resolve_hdfs_name_service('foobar')
    # None for nonexistent nameservice (intentional design)
    assert suj.resolve_hdfs_name_service('nonexistent') is None


def test_env_hadoop_home_prefix_install(mock_hadoop_home_directory):
    # The second+third env vars won't cause an error
    with mock.patch.dict(os.environ, {'HADOOP_PREFIX': '{}/no/where/here'.format(mock_hadoop_home_directory),
                                      'HADOOP_INSTALL': '{}/no/where/here'.format(mock_hadoop_home_directory),
                                      'HADOOP_HOME': mock_hadoop_home_directory}, clear=True):
        _test_default_hdfs_service(mock_hadoop_home_directory, 'HADOOP_HOME')


def test_env_hadoop_prefix_only(mock_hadoop_home_directory):
    with mock.patch.dict(os.environ, {'HADOOP_PREFIX': mock_hadoop_home_directory}, clear=True):
        _test_default_hdfs_service(mock_hadoop_home_directory, 'HADOOP_PREFIX')


def test_env_hadoop_install_only(mock_hadoop_home_directory):
    with mock.patch.dict(os.environ, {'HADOOP_INSTALL': mock_hadoop_home_directory}, clear=True):
        _test_default_hdfs_service(mock_hadoop_home_directory, 'HADOOP_INSTALL')


def test_env_bad_hadoop_home_with_hadoop_install(mock_hadoop_home_directory):
    with mock.patch.dict(os.environ, {'HADOOP_HOME': '{}/no/where/here'.format(mock_hadoop_home_directory),
                                      'HADOOP_INSTALL': mock_hadoop_home_directory}, clear=True):
        with pytest.raises(IOError):
            # Trigger env var evaluation
            HdfsNamenodeResolver()


def test_unmatched_env_var(mock_hadoop_home_directory):
    with mock.patch.dict(os.environ, {'HADOOP_HOME_X': mock_hadoop_home_directory}, clear=True):
        suj = HdfsNamenodeResolver()
        # No successful connection
        with pytest.raises(RuntimeError):
            suj.resolve_default_hdfs_service()


def test_bad_hadoop_path(mock_hadoop_home_directory):
    with mock.patch.dict(os.environ, {'HADOOP_HOME': '{}/no/where/here'.format(mock_hadoop_home_directory)},
                         clear=True):
        with pytest.raises(IOError):
            HdfsNamenodeResolver()


def test_missing_or_empty_core_site(mock_hadoop_home_directory):
    with mock.patch.dict(os.environ, {'HADOOP_HOME': mock_hadoop_home_directory}):
        # Make core-site "disappear" and make sure we raise an error
        cur_path = '{}{}/core-site.xml'.format(mock_hadoop_home_directory, HC.HADOOP_CONFIG_PATH)
        new_path = '{}{}/core-site.xml.bak'.format(mock_hadoop_home_directory, HC.HADOOP_CONFIG_PATH)
        os.rename(cur_path, new_path)
        with pytest.raises(IOError):
            HdfsNamenodeResolver()
        # Make an empty file
        with open(cur_path, 'wt') as f:
            f.write('')
        # Re-trigger env var evaluation
        suj = HdfsNamenodeResolver()
        with pytest.raises(RuntimeError):
            suj.resolve_default_hdfs_service()
        # restore file for other tests to work
        os.rename(new_path, cur_path)


class HdfsMockError(Exception):
    pass


class MockHdfs(object):
    """
    Any operation in the mock class raises an exception for the first N failovers, and then returns
    True after those N calls.
    """

    def __init__(self, n_failovers=0):
        self._n_failovers = n_failovers

    def __getattribute__(self, attr):
        """
        The Mock HDFS simply calls check_failover, regardless of the filesystem operator invoked.
        """

        def op(*args, **kwargs):
            """ Mock operator """
            return self._check_failovers()

        # Of course, exclude any protected/private method calls
        if not attr.startswith('_'):
            return op
        return object.__getattribute__(self, attr)

    def _check_failovers(self):
        if self._n_failovers == -1:
            # Special case to exercise the unhandled exception path
            raise HdfsMockError('Some random HDFS exception!')

        if self._n_failovers > 0:
            self._n_failovers -= 1
            raise ArrowIOError('org.apache.hadoop.ipc.RemoteException'
                               '(org.apache.hadoop.ipc.StandbyException): '
                               'Operation category READ is not supported in state standby. '
                               'Visit https://s.apache.org/sbnn-error\n'
                               '{} namenode failover(s) remaining!'.format(self._n_failovers))
        return True


class MockHdfsConnector(HdfsConnector):
    # static member for static hdfs_connect_namenode to access
    _n_failovers = 0
    _fail_n_next_connect = 0
    _connect_attempted = {}

    @classmethod
    def reset(cls):
        cls._n_failovers = 0
        cls._fail_n_next_connect = 0
        cls._connect_attempted = {}

    @classmethod
    def set_n_failovers(cls, failovers):
        cls._n_failovers = failovers

    @classmethod
    def set_fail_n_next_connect(cls, fails):
        cls._fail_n_next_connect = fails

    @classmethod
    def connect_attempted(cls, host):
        if host in cls._connect_attempted:
            return cls._connect_attempted[host]
        else:
            return 0

    @classmethod
    def hdfs_connect_namenode(cls, url, driver='libhdfs3'):
        netloc = '{}:{}'.format(url.hostname or 'default', url.port or 8020)
        if netloc not in cls._connect_attempted:
            cls._connect_attempted[netloc] = 0
        cls._connect_attempted[netloc] += 1
        # We just want to check connection attempt, but also raise an error if
        # 'default' or fail counter
        if cls._fail_n_next_connect != 0 or netloc == HC.DEFAULT_NN:
            if cls._fail_n_next_connect != 0:
                cls._fail_n_next_connect -= 1
            raise ArrowIOError('ERROR! Mock pyarrow hdfs connect to {} using driver {}, '
                               'fail counter: {}'
                               .format(netloc, driver, cls._fail_n_next_connect))
        # Return a mock HDFS object with optional failovers, so that this connector mock can
        # be shared for the HAHdfsClient failover tests below.
        hdfs = MockHdfs(cls._n_failovers)
        if cls._n_failovers > 0:
            cls._n_failovers -= 1
        return hdfs


class HdfsConnectorTest(unittest.TestCase):
    """Check correctness of connecting to a list of namenodes. """

    @classmethod
    def setUpClass(cls):
        """Initializes a mock HDFS namenode connector to track connection attempts."""
        cls.NAMENODES = [HC.WARP_TURTLE_NN1, HC.WARP_TURTLE_NN2]
        cls.suj = MockHdfsConnector()

    def setUp(self):
        self.suj.reset()

    def test_connect_to_either_namenode_ok(self):
        """ Test connecting OK to first of name node URLs. """
        self.assertIsNotNone(self.suj.connect_to_either_namenode(self.NAMENODES))
        self.assertEqual(0, self.suj.connect_attempted(HC.DEFAULT_NN))
        self.assertEqual(1, self.suj.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(0, self.suj.connect_attempted(HC.WARP_TURTLE_NN2))

    def test_connect_to_either_namenode_ok_one_failed(self):
        """ With one failver, test that both namenode URLS are attempted, with 2nd connected. """
        self.suj.set_fail_n_next_connect(1)
        self.assertIsNotNone(self.suj.connect_to_either_namenode(self.NAMENODES))
        self.assertEqual(0, self.suj.connect_attempted(HC.DEFAULT_NN))
        self.assertEqual(1, self.suj.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(1, self.suj.connect_attempted(HC.WARP_TURTLE_NN2))

    def test_connect_to_either_namenode_exception_two_failed(self):
        """ With 2 failvers, test no connection, and no exception is raised. """
        self.suj.set_fail_n_next_connect(2)
        with self.assertRaises(HdfsConnectError):
            self.suj.connect_to_either_namenode(self.NAMENODES)
        self.assertEqual(0, self.suj.connect_attempted(HC.DEFAULT_NN))
        self.assertEqual(1, self.suj.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(1, self.suj.connect_attempted(HC.WARP_TURTLE_NN2))

    def test_connect_to_either_namenode_exception_four_failed(self):
        """ With 4 failvers, test that exception is raised. """
        self.suj.set_fail_n_next_connect(4)
        with self.assertRaises(HdfsConnectError):
            self.suj.connect_to_either_namenode(self.NAMENODES)
        with self.assertRaises(HdfsConnectError):
            self.suj.connect_to_either_namenode(self.NAMENODES)
        self.assertEqual(0, self.suj.connect_attempted(HC.DEFAULT_NN))
        self.assertEqual(2, self.suj.connect_attempted(HC.WARP_TURTLE_NN1))
        self.assertEqual(2, self.suj.connect_attempted(HC.WARP_TURTLE_NN2))


class HAHdfsClientTest(unittest.TestCase):
    """
    The HDFS testing functions are enumerated explicitly below for simplicity and clarity, but it
    should impose but a minute maintenance overhead, since MockHdfs class requires no enumeration.
    """

    @classmethod
    def setUpClass(cls):
        """Initializes namenodes list and mock HDFS namenode connector."""
        cls.NAMENODES = [HC.WARP_TURTLE_NN1, HC.WARP_TURTLE_NN2]

    def setUp(self):
        """Reset mock HDFS failover count."""
        MockHdfsConnector.reset()

    def test_unhandled_exception(self):
        """Exercise the unhandled exception execution path."""
        MockHdfsConnector.set_n_failovers(-1)
        with self.assertRaises(HdfsMockError) as e:
            getattr(HAHdfsClient(MockHdfsConnector, [HC.WARP_TURTLE_NN1]), 'ls')('random')
        self.assertTrue('random HDFS exception' in str(e.exception))

    def test_invalid_namenode_list(self):
        """Make sure robust to invalid namenode list."""
        MockHdfsConnector.set_n_failovers(-1)
        with self.assertRaises(HdfsConnectError) as e:
            getattr(HAHdfsClient(MockHdfsConnector, []), 'ls')('random')
        self.assertTrue('Unable to connect' in str(e.exception))
        with self.assertRaises(HdfsConnectError) as e:
            getattr(HAHdfsClient(MockHdfsConnector, [None]), 'ls')('random')
        self.assertTrue('Unable to connect' in str(e.exception))

    def test_client_pickles_correctly(self):
        """
        Does HAHdfsClient pickle properly?

        Check that all attributes are equal, with the exception of the HDFS object, which is fine
        as long as the types are the same.
        """
        client = HAHdfsClient(MockHdfsConnector, self.NAMENODES)
        client_unpickled = pickle.loads(pickle.dumps(client))
        self.assertEqual(client._connector_cls, client_unpickled._connector_cls)
        self.assertEqual(client._list_of_namenodes, client_unpickled._list_of_namenodes)
        self.assertEqual(client._index_of_nn, client_unpickled._index_of_nn)
        self.assertEqual(type(client._hdfs), type(client_unpickled._hdfs))

    def _try_failover_combos(self, func, *args, **kwargs):
        """Common tests for each of the known HDFS operators, with varying failover counts."""
        MockHdfsConnector.set_n_failovers(1)
        suj = HAHdfsClient(MockHdfsConnector, self.NAMENODES)
        self.assertTrue(getattr(suj, func)(*args, **kwargs))

        MockHdfsConnector.set_n_failovers(namenode_failover.MAX_FAILOVER_ATTEMPTS)
        suj = HAHdfsClient(MockHdfsConnector, self.NAMENODES)
        self.assertTrue(getattr(suj, func)(*args, **kwargs))

        MockHdfsConnector.set_n_failovers(namenode_failover.MAX_FAILOVER_ATTEMPTS + 1)
        suj = HAHdfsClient(MockHdfsConnector, self.NAMENODES)
        with self.assertRaises(MaxFailoversExceeded) as e:
            getattr(suj, func)(*args, **kwargs)
        self.assertEqual(len(e.exception.failed_exceptions),
                         namenode_failover.MAX_FAILOVER_ATTEMPTS + 1)
        self.assertEqual(e.exception.max_failover_attempts, namenode_failover.MAX_FAILOVER_ATTEMPTS)
        self.assertEqual(e.exception.__name__, func)
        self.assertTrue('Failover attempts exceeded' in str(e.exception))

    def test_cat(self):
        self._try_failover_combos('cat', 'random')

    def test_chmod(self):
        self._try_failover_combos('chmod', 'random', 0)

    def test_chown(self):
        self._try_failover_combos('chown', 'random', 'user')

    def test_delete(self):
        self._try_failover_combos('delete', 'random', recursive=True)

    def test_df(self):
        self._try_failover_combos('df')

    def test_disk_usage(self):
        self._try_failover_combos('disk_usage', 'random')

    def test_download(self):
        self._try_failover_combos('download', 'random', None)

    def test_exists(self):
        self._try_failover_combos('exists', 'random')

    def test_get_capacity(self):
        self._try_failover_combos('get_capacity')

    def test_get_space_used(self):
        self._try_failover_combos('get_space_used')

    def test_info(self):
        self._try_failover_combos('info', 'random')

    def test_ls(self):
        self._try_failover_combos('ls', 'random', detail=True)

    def test_mkdir(self):
        self._try_failover_combos('mkdir', 'random', create_parents=False)

    def test_open(self):
        self._try_failover_combos('open', 'random', 'rb')

    def test_rename(self):
        self._try_failover_combos('rename', 'random', 'new_random')

    def test_rm(self):
        self._try_failover_combos('rm', 'random', recursive=True)

    def test_upload(self):
        self._try_failover_combos('upload', 'random', None)


if __name__ == '__main__':
    unittest.main()
