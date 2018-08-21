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
import os
import pickle
from base64 import b64encode, b64decode
from collections import namedtuple

import pytest
import six

from petastorm.tests.test_common import create_test_dataset

SyntheticDataset = namedtuple('synthetic_dataset', ['url', 'data', 'path'])

# Number of rows in a fake dataset

ROWS_COUNT = 100

_CACHE_FAKE_DATASET_OPTION = '--cache-synthetic-dataset'

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption(
        _CACHE_FAKE_DATASET_OPTION, action="store_true", default=False,
        help='Use a cached version of synthetic dataset if available. This helps speedup local tests reruns as '
             'we don\'t have to rerun spark. CAUTION: you won\'t be exercising dataset generating parts of petastorm '
             'hence tests results maybe inaccurate'
    )


@pytest.fixture(scope="session")
def synthetic_dataset(request, tmpdir_factory):
    # We speedup test startup time by caching previously generated synthetic dataset.
    # This is useful while developing for tests reruns, but can be dangerous since we can
    # get stale results when petastorm code participating in dataset generation is used.
    if request.config.getoption(_CACHE_FAKE_DATASET_OPTION):
        cache_key = 'synthetic_dataset_{}'.format('PY2' if six.PY2 else 'PY3')
        serialized = request.config.cache.get(cache_key, None)
        dataset = pickle.loads(b64decode(serialized)) if serialized else None
        if not dataset or not os.path.exists(dataset.path):
            dataset = _synthetic_dataset_no_cache(tmpdir_factory)
            request.config.cache.set(cache_key, b64encode(pickle.dumps(dataset)).decode('ascii'))
        else:
            logger.warn('CAUTION: %s HAS BEEN USED. CACHED TEST DATASET! MAYBE STALE!', _CACHE_FAKE_DATASET_OPTION)
    else:
        dataset = _synthetic_dataset_no_cache(tmpdir_factory)

    return dataset


def _synthetic_dataset_no_cache(tmpdir_factory):
    path = tmpdir_factory.mktemp("data").strpath
    url = 'file://' + path
    data = create_test_dataset(url, range(ROWS_COUNT))
    dataset = SyntheticDataset(url=url, path=path, data=data)
    return dataset
