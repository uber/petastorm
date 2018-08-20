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

import pyarrow
import six
from six.moves.urllib.parse import urlparse

from petastorm.hdfs.namenode import HdfsNamenodeResolver, HdfsConnector


class FilesystemResolver(object):
    """Resolves a dataset URL, makes a connection via pyarrow, and provides a filesystem object."""

    def __init__(self, dataset_url, hadoop_configuration=None, connector=HdfsConnector):
        """
        Given a dataset URL and an optional hadoop configuration, parse and interpret the URL to
        instantiate a pyarrow filesystem.

        Interpretation of the URL ``scheme://hostname:port/path`` occurs in the following order:

        1. If no ``scheme``, no longer supported, so raise an exception!
        2. If ``scheme`` is ``file``, use local filesystem path.
        3. If ``scheme`` is ``hdfs``:
           a. Try the ``hostname`` as a namespace and attempt to connect to a name node.
           b. If no host, connect to the default name node.
        4. Next, try connecting directly to namenode ``hostname:port``.
        5. Fail otherwise.

        :param dataset_url: The hdfs URL or absolute path to the dataset
        :param hadoop_configuration: an optional hadoop configuration
        :param connector: the HDFS connector object to use (ONLY override for testing purposes)
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

        elif self._parsed_dataset_url.scheme == 'hdfs':
            # Obtain singleton and force hadoop config evaluation
            namenode_resolver = HdfsNamenodeResolver(hadoop_configuration)

            # Since we can't tell for sure, first treat the URL as though it references a name service
            if self._parsed_dataset_url.netloc:
                # Case 3a: Use the portion of netloc before any port, which doesn't get lowercased
                nameservice = self._parsed_dataset_url.netloc.split(':')[0]
                namenodes = namenode_resolver.resolve_hdfs_name_service(nameservice)
                if namenodes:
                    self._filesystem = connector.connect_to_either_namenode(namenodes)
                if self._filesystem is None:
                    # Case 5: That didn't work; try the URL as a namenode host
                    self._filesystem = connector.hdfs_connect_namenode(self._parsed_dataset_url)
            else:
                # Case 3b: No netloc, so let's try to connect to default namenode
                # HdfsNamenodeResolver will raise exception if it fails to connect.
                nameservice, namenodes = namenode_resolver.resolve_default_hdfs_service()
                filesystem = connector.connect_to_either_namenode(namenodes)
                if filesystem is not None:
                    # Properly replace the parsed dataset URL once default namenode is confirmed
                    self._parsed_dataset_url = urlparse(
                        'hdfs://{}{}'.format(nameservice, self._parsed_dataset_url.path))
                    self._filesystem = filesystem

        else:
            # Case 5
            raise ValueError('Unsupported scheme in dataset url {}. '
                             'Currently, only "file" and "hdfs" are supported.'.format(self._parsed_dataset_url.scheme))

    def parsed_dataset_url(self):
        """
        :return: The urlparse'd dataset_url
        """
        return self._parsed_dataset_url

    def filesystem(self):
        """
        :return: The pyarrow filesystem object
        """
        return self._filesystem
