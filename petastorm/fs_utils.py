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
import logging

import fsspec
import pyarrow
import six
from six.moves.urllib.parse import urlparse, urlunparse
from fsspec.core import strip_protocol
from fsspec.utils import update_storage_options

from petastorm.hdfs.namenode import HdfsNamenodeResolver, HdfsConnector

logger = logging.getLogger(__name__)


def get_dataset_path(parsed_url):
    """
    The dataset path is different than the one in `_parsed_dataset_url` for some filesystems.
    For example s3fs expects the bucket name to be included in the path and doesn't support
    paths that start with a `/`
    """
    if parsed_url.scheme.lower() in ['file', 'hdfs']:
        return parsed_url.path

    _url = strip_protocol(urlunparse(parsed_url))
    return _url


class FilesystemResolver(object):
    """Resolves a dataset URL, makes a connection via pyarrow, and provides a filesystem object."""

    def __init__(self, dataset_url, hadoop_configuration=None, connector=HdfsConnector,
                 hdfs_driver='libhdfs3', user=None, storage_options=None):
        """
        Given a dataset URL and an optional hadoop configuration, parse and interpret the URL to
        instantiate a pyarrow filesystem.

        Interpretation of the URL ``scheme://hostname:port/path`` occurs in the following order:

        1. If no ``scheme``, no longer supported, so raise an exception!
        2. If ``scheme`` is ``file``, use local filesystem path.
        3. If ``scheme`` is ``hdfs``:
           a. Try the ``hostname`` as a namespace and attempt to connect to a name node.
              1. If that doesn't work, try connecting directly to namenode ``hostname:port``.
           b. If no host, connect to the default name node.
        5. If ``scheme`` is ``s3``, use s3fs. The user must manually install s3fs before using s3
        6. If ``scheme`` is ``gs``or ``gcs``, use gcsfs. The user must manually install gcsfs before using GCS
        7. Fail otherwise.

        :param dataset_url: The hdfs URL or absolute path to the dataset
        :param hadoop_configuration: an optional hadoop configuration
        :param connector: the HDFS connector object to use (ONLY override for testing purposes)
        :param hdfs_driver: A string denoting the hdfs driver to use (if using a dataset on hdfs). Current choices are
        libhdfs (java through JNI) or libhdfs3 (C++)
        :param user: String denoting username when connecting to HDFS. None implies login user.
        :param storage_options: Dict of kwargs forwarded to ``fsspec`` to initialize the filesystem.
        """
        # Cache both the original URL and the resolved, urlparsed dataset_url
        self._dataset_url = dataset_url
        self._parsed_dataset_url = None
        # Cache the instantiated filesystem object
        self._filesystem = None

        if isinstance(self._dataset_url, six.string_types):
            self._parsed_dataset_url = urlparse(self._dataset_url)
        else:
            self._parsed_dataset_url = self._dataset_url

        if not self._parsed_dataset_url.scheme:
            # Case 1
            raise ValueError('ERROR! A scheme-less dataset url ({}) is no longer supported. '
                             'Please prepend "file://" for local filesystem.'.format(self._parsed_dataset_url.scheme))

        elif self._parsed_dataset_url.scheme == 'file':
            # Case 2: definitely local
            self._filesystem = pyarrow.localfs
            self._filesystem_factory = lambda: pyarrow.localfs

        elif self._parsed_dataset_url.scheme == 'hdfs':

            if hdfs_driver == 'libhdfs3':
                # libhdfs3 does not do any namenode resolution itself so we do it manually. This is not necessary
                # if using libhdfs

                # Obtain singleton and force hadoop config evaluation
                namenode_resolver = HdfsNamenodeResolver(hadoop_configuration)

                # Since we can't tell for sure, first treat the URL as though it references a name service
                if self._parsed_dataset_url.netloc:
                    # Case 3a: Use the portion of netloc before any port, which doesn't get lowercased
                    nameservice = self._parsed_dataset_url.netloc.split(':')[0]
                    namenodes = namenode_resolver.resolve_hdfs_name_service(nameservice)
                    if namenodes:
                        self._filesystem = connector.connect_to_either_namenode(namenodes, user=user)
                        self._filesystem_factory = lambda: connector.connect_to_either_namenode(namenodes, user=user)
                    if self._filesystem is None:
                        # Case 3a1: That didn't work; try the URL as a namenode host
                        self._filesystem = connector.hdfs_connect_namenode(self._parsed_dataset_url, user=user)
                        self._filesystem_factory = \
                            lambda url=self._dataset_url, user=user: \
                            connector.hdfs_connect_namenode(urlparse(url), user=user)
                else:
                    # Case 3b: No netloc, so let's try to connect to default namenode
                    # HdfsNamenodeResolver will raise exception if it fails to connect.
                    nameservice, namenodes = namenode_resolver.resolve_default_hdfs_service()
                    filesystem = connector.connect_to_either_namenode(namenodes, user=user)
                    self._filesystem_factory = lambda: connector.connect_to_either_namenode(namenodes, user=user)
                    if filesystem is not None:
                        # Properly replace the parsed dataset URL once default namenode is confirmed
                        self._parsed_dataset_url = urlparse(
                            'hdfs://{}{}'.format(nameservice, self._parsed_dataset_url.path))
                        self._filesystem = filesystem
            else:
                self._filesystem = connector.hdfs_connect_namenode(self._parsed_dataset_url, hdfs_driver, user=user)
                self._filesystem_factory = \
                    lambda url=self._dataset_url, user=user: \
                    connector.hdfs_connect_namenode(urlparse(url), hdfs_driver, user=user)
        else:
            # Fallback to fsspec to handle any other schemes
            if not self._parsed_dataset_url.netloc:
                raise ValueError(
                    'URLs must be of the form {}://bucket/path'.format(self._parsed_dataset_url.scheme)
                )

            storage_options = storage_options or {}
            protocol = self._parsed_dataset_url.scheme
            cls = fsspec.get_filesystem_class(protocol)
            options = cls._get_kwargs_from_urls(self._dataset_url)
            update_storage_options(options, storage_options)
            self._filesystem = cls(**options)
            self._filesystem_factory = lambda: cls(**options)  # pylint: disable=unnecessary-lambda

    def parsed_dataset_url(self):
        """
        :return: The urlparse'd dataset_url
        """
        return self._parsed_dataset_url

    def get_dataset_path(self):
        """
        The dataset path is different than the one in `_parsed_dataset_url` for some filesystems.
        For example s3fs expects the bucket name to be included in the path and doesn't support
        paths that start with a `/`
        """
        return get_dataset_path(self._parsed_dataset_url)

    def filesystem(self):
        """
        :return: The pyarrow filesystem object
        """
        return self._filesystem

    def filesystem_factory(self):
        """
        :return: A serializable function that can be used to recreate the filesystem
        object; useful for providing access to the filesystem object on distributed
        Spark executors.
        """
        return self._filesystem_factory

    def __getstate__(self):
        raise RuntimeError('Pickling FilesystemResolver is not supported as it may contain some '
                           'a file-system instance objects that do not support pickling but do not have '
                           'anti-pickling protection')


def get_filesystem_and_path_or_paths(url_or_urls, hdfs_driver='libhdfs3', storage_options=None, filesystem=None):
    """
    Given a url or url list, return a tuple ``(filesystem, path_or_paths)``
    ``filesystem`` is created from the given url(s), and ``path_or_paths`` is a path or path list
    extracted from the given url(s)
    if url list given, the urls must have the same scheme and netloc.
    """
    if isinstance(url_or_urls, list):
        url_list = url_or_urls
    else:
        url_list = [url_or_urls]

    parsed_url_list = [urlparse(url) for url in url_list]

    first_scheme = parsed_url_list[0].scheme
    first_netloc = parsed_url_list[0].netloc

    for parsed_url in parsed_url_list:
        if parsed_url.scheme != first_scheme or parsed_url.netloc != first_netloc:
            raise ValueError('The dataset url list must contain url with the same scheme and netloc.')

    fs = filesystem or FilesystemResolver(
        url_list[0], hdfs_driver=hdfs_driver, storage_options=storage_options).filesystem()
    path_list = [get_dataset_path(parsed_url) for parsed_url in parsed_url_list]

    if isinstance(url_or_urls, list):
        path_or_paths = path_list
    else:
        path_or_paths = path_list[0]

    return fs, path_or_paths


def normalize_dir_url(dataset_url):
    if dataset_url is None or not isinstance(dataset_url, six.string_types):
        raise ValueError('directory url must be a string')

    dataset_url = dataset_url[:-1] if dataset_url[-1] == '/' else dataset_url
    logger.debug('directory url: %s', dataset_url)
    return dataset_url
