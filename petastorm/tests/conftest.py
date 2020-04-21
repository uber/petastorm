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
import tempfile
from base64 import b64encode, b64decode
from collections import namedtuple

import pytest
import six

from petastorm.spark import spark_dataset_converter, SparkDatasetConverter
from petastorm.tests.test_common import create_test_dataset, create_test_scalar_dataset, \
    create_many_columns_non_petastorm_dataset
from pyspark.sql import SparkSession

SyntheticDataset = namedtuple('synthetic_dataset', ['url', 'data', 'path'])

# Number of rows in a fake dataset
_ROWS_COUNT = 100

_CACHE_FAKE_DATASET_OPTION_SHORT = '-Y'
_CACHE_FAKE_DATASET_OPTION = '--cache-synthetic-dataset'

logger = logging.getLogger(__name__)


def pytest_logger_config(logger_config):
    logger_config.add_loggers(
        [
            'petastorm.workers_pool.process_pool',
            'petastorm.workers_pool.thread_pool',
            'petastorm.workers_pool.dummy_pool',
            'petastorm.workers_pool.ventilator',
            'petastorm.reader',
        ], stdout_level='debug')


def pytest_addoption(parser):
    parser.addoption(
        _CACHE_FAKE_DATASET_OPTION_SHORT, _CACHE_FAKE_DATASET_OPTION, action="store_true", default=False,
        help='Use a cached version of synthetic dataset if available. This helps speedup local tests reruns as '
             'we don\'t have to rerun spark. CAUTION: you won\'t be exercising dataset generating parts of petastorm '
             'hence tests results maybe inaccurate'
    )


def maybe_cached_dataset(config, name, generating_func):
    """Returns cached dataset instance if caching of datasets is enabled and a valid dataset is available.

    We speedup test startup time by caching previously generated synthetic dataset.
    This is useful while developing for tests reruns, but can be dangerous since we can
    get stale results when petastorm code participating in dataset generation is used.

    :param config: request.config object.
    :param name: name of the cached dataset. Used as a cache key.
    :param generating_func: This function will be called (`generating_func()`) if dataset cache is disabled or
      no valid dataset is found in cache.
    :return:
    """
    if config.getoption(_CACHE_FAKE_DATASET_OPTION):
        cache_key = '{}_{}'.format(name, 'PY2' if six.PY2 else 'PY3')
        serialized = config.cache.get(cache_key, None)
        dataset = pickle.loads(b64decode(serialized)) if serialized else None
        if not dataset or not os.path.exists(dataset.path):
            dataset = generating_func()
            config.cache.set(cache_key, b64encode(pickle.dumps(dataset)).decode('ascii'))
        else:
            logger.warning('CAUTION: %s HAS BEEN USED. USING %s CACHED TEST DATASET! MAYBE STALE!',
                           _CACHE_FAKE_DATASET_OPTION, name)
    else:
        dataset = generating_func()

    return dataset


@pytest.fixture(scope="session")
def synthetic_dataset(request, tmpdir_factory):
    def _synthetic_dataset_no_cache():
        path = tmpdir_factory.mktemp("data").strpath
        url = 'file://' + path
        data = create_test_dataset(url, range(_ROWS_COUNT))
        dataset = SyntheticDataset(url=url, path=path, data=data)
        return dataset

    return maybe_cached_dataset(request.config, 'synthetic_dataset', _synthetic_dataset_no_cache)


@pytest.fixture(scope="session")
def scalar_dataset(request, tmpdir_factory):
    def _pure_parquet_dataset_no_cache():
        path = tmpdir_factory.mktemp("data").strpath
        url = 'file://' + path
        data = create_test_scalar_dataset(url, 100)
        dataset = SyntheticDataset(url=url, path=path, data=data)
        return dataset

    return maybe_cached_dataset(request.config, 'scalar', _pure_parquet_dataset_no_cache)


@pytest.fixture(scope="session")
def many_columns_non_petastorm_dataset(request, tmpdir_factory):
    """This dataset has 1000 columns. All of the same int32 type."""

    def _dataset_no_cache():
        path = tmpdir_factory.mktemp("data").strpath
        url = 'file://' + path
        data = create_many_columns_non_petastorm_dataset(url, 10)
        dataset = SyntheticDataset(url=url, path=path, data=data)
        return dataset

    return maybe_cached_dataset(request.config, 'many_column_non_petastorm', _dataset_no_cache)


class SparkTestContext(object):
    def __init__(self):
        self.spark = SparkSession.builder \
            .master("local[2]") \
            .appName("petastorm.spark tests") \
            .getOrCreate()
        self.tempdir = tempfile.mkdtemp('_spark_converter_test')
        self.temp_url = 'file://' + self.tempdir.replace(os.sep, '/')
        self.spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, self.temp_url)
        spark_dataset_converter._FILE_AVAILABILITY_WAIT_TIMEOUT_SECS = 2

    def tear_down(self):
        # restore default file availability wait timeout
        spark_dataset_converter._FILE_AVAILABILITY_WAIT_TIMEOUT_SECS = 30
        self.spark.stop()


@pytest.fixture(scope='module')
def spark_test_ctx():
    ctx = SparkTestContext()
    try:
        yield ctx
    finally:
        ctx.tear_down()
