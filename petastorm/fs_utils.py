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

import pyarrow
import six
from six.moves.urllib.parse import urlparse

logger = logging.getLogger(__name__)


def get_dataset_path(parsed_url):
    """
    The dataset path is different than the one in `_parsed_dataset_url` for some filesystems.
    For example s3fs expects the bucket name to be included in the path and doesn't support
    paths that start with a `/`
    """
    if parsed_url.scheme.lower() in ['s3', 's3a', 's3n', 'gs', 'gcs']:
        # s3/gs/gcs filesystem expects paths of the form `bucket/path`
        return parsed_url.netloc + parsed_url.path

    return parsed_url.path


def get_filesystem_and_path_or_paths(url_or_urls, hdfs_driver='libhdfs3', s3_config_kwargs=None):
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

    fs, _ = pyarrow.fs.FileSystem.from_uri(url_list[0])
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
