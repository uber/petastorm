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

import functools
import inspect
import logging
import os
from distutils.version import LooseVersion
from xml.etree import ElementTree as ET

import pyarrow
import six
from pyarrow.hdfs import HadoopFileSystem
from pyarrow.lib import ArrowIOError
from six.moves.urllib.parse import urlparse

logger = logging.getLogger(__name__)


class HdfsNamenodeResolver(object):
    """This class embodies functionality to resolve HDFS namenodes: per default or a nameservice."""

    def __init__(self, hadoop_configuration=None):
        """
        Sets the given HadoopConfiguration object for the resolver; or check for and pull hadoop
        configuration from an environment variable, in below preferred order to check.

        :param hadoop_configuration: an optional ``HadoopConfiguration``
        """
        self._hadoop_env = None
        self._hadoop_path = None
        if hadoop_configuration is None:
            # Pull from environment variable, in this preferred order
            for env in ["HADOOP_HOME", "HADOOP_PREFIX", "HADOOP_INSTALL"]:
                # Use the first available
                if env in os.environ:
                    self._hadoop_env = env
                    self._hadoop_path = os.environ[env]
                    hadoop_configuration = {}
                    self._load_site_xml_into_dict(
                        '{}/etc/hadoop/hdfs-site.xml'.format(self._hadoop_path),
                        hadoop_configuration)
                    self._load_site_xml_into_dict(
                        '{}/etc/hadoop/core-site.xml'.format(self._hadoop_path),
                        hadoop_configuration)
                    break
            if hadoop_configuration is None:
                # ensures at least an empty dict so no further checks required in member functions
                logger.warning('Unable to populate a sensible HadoopConfiguration for namenode resolution!\n'
                               'Path of last environment var (%s) tried [%s]. Please set up your Hadoop and \n'
                               'define environment variable HADOOP_HOME to point to your Hadoop installation path.',
                               self._hadoop_env, self._hadoop_path)
                hadoop_configuration = {}
        self._hadoop_configuration = hadoop_configuration

    def _load_site_xml_into_dict(self, xml_path, in_dict):
        assert in_dict is not None, 'A valid dictionary must be supplied to process site XML'
        try:
            for prop in ET.parse(xml_path).getroot().iter('property'):
                in_dict[prop.find('name').text] = prop.find('value').text
        except ET.ParseError as ex:
            logger.error(
                'Unable to obtain a root node for the supplied XML in %s: %s', xml_path, ex)

    def _build_error_string(self, msg):
        if self._hadoop_path is not None:
            return msg + '\nHadoop path {} in environment variable {}!\n' \
                         'Please check your hadoop configuration!' \
                .format(self._hadoop_path, self._hadoop_env)
        else:
            return msg + ' the supplied Spark HadoopConfiguration'

    def resolve_hdfs_name_service(self, namespace):
        """
        Given the namespace of a name service, resolves the configured list of name nodes, and
        returns them as a list of URL strings.

        :param namespace: the HDFS name service to resolve
        :return: a list of URL strings of the name nodes for the given name service; or None of not
            properly configured.
        """
        list_of_namenodes = None
        namenodes = self._hadoop_configuration.get('dfs.ha.namenodes.' + namespace)
        if namenodes:
            # populate namenode_urls list for the given namespace
            list_of_namenodes = []
            for nn in namenodes.split(','):
                prop_key = 'dfs.namenode.rpc-address.{}.{}'.format(namespace, nn)
                namenode_url = self._hadoop_configuration.get(prop_key)
                if namenode_url:
                    list_of_namenodes.append(namenode_url)
                else:
                    raise RuntimeError(self._build_error_string('Failed to get property "{}" from'
                                                                .format(prop_key)))
        # Don't raise and exception otherwise, because the supplied name could just be a hostname.
        # We don't have an easy way to tell at this point.
        return list_of_namenodes

    def resolve_default_hdfs_service(self):
        """
        Resolves the default namenode using the given, or environment-derived, hadoop configuration,
        by parsing the configuration for ``fs.defaultFS``.

        :return: a tuple of structure ``(nameservice, list of namenodes)``
        """
        default_fs = self._hadoop_configuration.get('fs.defaultFS')
        if default_fs:
            nameservice = urlparse(default_fs).netloc
            list_of_namenodes = self.resolve_hdfs_name_service(nameservice)
            if list_of_namenodes is None:
                raise IOError(self._build_error_string('Unable to get namenodes for '
                                                       'default service "{}" from'
                                                       .format(default_fs)))
            return [nameservice, list_of_namenodes]
        else:
            raise RuntimeError(
                self._build_error_string('Failed to get property "fs.defaultFS" from'))


class HdfsConnectError(IOError):
    pass


class MaxFailoversExceeded(RuntimeError):
    def __init__(self, failed_exceptions, max_failover_attempts, func_name):
        self.failed_exceptions = failed_exceptions
        self.max_failover_attempts = max_failover_attempts
        self.__name__ = func_name
        message = 'Failover attempts exceeded maximum ({}) for action "{}". ' \
                  'Exceptions:\n{}'.format(self.max_failover_attempts, self.__name__,
                                           self.failed_exceptions)
        super(MaxFailoversExceeded, self).__init__(message)


class namenode_failover(object):
    """
    This decorator class ensures seamless namenode failover and retry, when an HDFS call fails
    due to StandbyException, up to a maximum retry.
    """
    # Allow for 2 failovers to a different namenode (i.e., if 2 NNs, try back to the original)
    MAX_FAILOVER_ATTEMPTS = 2

    def __init__(self, func):
        # limit wrapper attributes updated to just name and doc string
        functools.update_wrapper(self, func, ('__name__', '__doc__'))
        # cache the function name, only because we don't need the function object in __call__
        self._func_name = func.__name__

    def __get__(self, obj, obj_type):
        """ Support usage of decorator on instance methods. """
        # This avoids needing to cache the `obj` as member variable
        return functools.partial(self.__call__, obj)

    def __call__(self, obj, *args, **kwargs):
        """
        Attempts the function call, catching exception, re-connecting, and retrying, up to a
        pre-configured maximum number of attempts.

        :param obj: calling class instance, the HDFS client object
        :param args: positional arguments to func
        :param kwargs: arbitrary keyword arguments to func
        :return: return of ``func`` call; if max retries exceeded, raise a RuntimeError; or raise
                any unexpected exception
        """
        failures = []
        while len(failures) <= self.MAX_FAILOVER_ATTEMPTS:
            try:
                # Invoke the filesystem function on the connected HDFS object
                return getattr(obj._hdfs, self._func_name)(*args, **kwargs)
            except ArrowIOError as e:
                # An HDFS IP error occurred, retry HDFS connect to failover
                obj._do_connect()
                failures.append(e)
        # Failover attempts exceeded at this point!
        raise MaxFailoversExceeded(failures, self.MAX_FAILOVER_ATTEMPTS, self._func_name)


def failover_all_class_methods(decorator):
    """
    This decorator function wraps an entire class to decorate each member method, incl. inherited.

    Adapted from https://stackoverflow.com/a/6307868
    """

    # Convenience function to ensure `decorate` gets wrapper function attributes: name, docs, etc.
    @functools.wraps(decorator)
    def decorate(cls):
        all_methods = inspect.getmembers(cls, inspect.isbuiltin) \
            + inspect.getmembers(cls, inspect.ismethod) \
            + inspect.getmembers(cls, inspect.isroutine)
        for name, method in all_methods:
            if not name.startswith('_'):
                # It's safer to exclude all protected/private method from decoration
                setattr(cls, name, decorator(method))
        return cls

    return decorate


@failover_all_class_methods(namenode_failover)
class HAHdfsClient(HadoopFileSystem):
    def __init__(self, connector_cls, list_of_namenodes):
        """
        Attempt HDFS connection operation, storing the hdfs object for intercepted calls.

        :param connector_cls: HdfsConnector class, so connector logic resides in one place, and
            also facilitates testing.
        :param list_of_namenodes: List of name nodes to failover, cached to enable un-/pickling
        """
        # Use protected attribute to prevent mistaken decorator application
        self._connector_cls = connector_cls
        self._list_of_namenodes = list_of_namenodes
        # Ensure that a retry will attempt a different name node in the list
        self._index_of_nn = -1
        self._do_connect()

    def __reduce__(self):
        """ Returns object state for pickling. """
        return self.__class__, (self._connector_cls, self._list_of_namenodes)

    def _do_connect(self):
        """ Makes a new connection attempt, caching the new namenode index and HDFS connection. """
        self._index_of_nn, self._hdfs = \
            self._connector_cls._try_next_namenode(self._index_of_nn, self._list_of_namenodes)


class HdfsConnector(object):
    """ HDFS connector class where failover logic is implemented.  Facilitates testing. """
    # Refactored constant
    MAX_NAMENODES = 2

    @classmethod
    def hdfs_connect_namenode(cls, url, driver='libhdfs3'):
        """
        Performs HDFS connect in one place, facilitating easy change of driver and test mocking.

        :param url: An parsed URL object to the HDFS end point
        :param driver: An optional driver identifier
        :return: Pyarrow HDFS connection object.
        """

        # According to pyarrow.hdfs.connect:
        #    host : NameNode. Set to "default" for fs.defaultFS from core-site.xml
        # So we pass 'default' as a host name if the url does not specify one (i.e. hdfs:///...)
        if LooseVersion(pyarrow.__version__) < LooseVersion('0.12.0'):
            hostname = url.hostname or 'default'
            driver = driver
        else:
            hostname = six.text_type(url.hostname or 'default')
            driver = six.text_type(driver)
        return pyarrow.hdfs.connect(hostname, url.port or 8020, driver=driver)

    @classmethod
    def connect_to_either_namenode(cls, list_of_namenodes):
        """
        Returns a wrapper HadoopFileSystem "high-availability client" object that enables
        name node failover.

        Raises a HdfsConnectError if no successful connection can be established.

        :param list_of_namenodes: a required list of name node URLs to connect to.
        :return: the wrapped HDFS connection object
        """
        assert list_of_namenodes is not None and len(list_of_namenodes) <= cls.MAX_NAMENODES, \
            "Must supply a list of namenodes, but HDFS only supports up to {} namenode URLs" \
            .format(cls.MAX_NAMENODES)
        return HAHdfsClient(cls, list_of_namenodes)

    @classmethod
    def _try_next_namenode(cls, index_of_nn, list_of_namenodes):
        """
        Instead of returning an inline function, this protected class method implements the
        failover logic: circling between namenodes using the supplied index as the last
        index into the name nodes list.

        :param list_of_namenodes: a required list of name node URLs to connect to.
        :return: a tuple of (new index into list, actual pyarrow HDFS connection object), or raise
                a HdfsConnectError if no successful connection can be established.
        """
        nn_len = len(list_of_namenodes)
        if nn_len > 0:
            for i in range(1, cls.MAX_NAMENODES + 1):
                # Use a modulo mechanism to hit the "next" name node, as opposed to always
                # starting from the first entry in the list
                idx = (index_of_nn + i) % nn_len
                host = list_of_namenodes[idx]
                try:
                    return idx, \
                        cls.hdfs_connect_namenode(urlparse('hdfs://' + str(host or 'default')))
                except ArrowIOError:
                    # This is an expected error if the namenode we are trying to connect to is
                    # not the active one
                    logger.debug('Attempted to connect to namenode %s but failed', host)
        # It is a problem if we cannot connect to either of the namenodes when tried back-to-back,
        # so better raise an error.
        raise HdfsConnectError("Unable to connect to HDFS cluster!")
