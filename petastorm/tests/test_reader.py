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
from time import sleep

import pyarrow.parquet as pq
import pytest

from petastorm import make_reader, TransformSpec
from petastorm.reader import Reader

# pylint: disable=unnecessary-lambda
READER_FACTORIES = [
    make_reader,
    lambda url, **kwargs: make_reader(url, reader_engine='experimental_reader_v2', **kwargs),
]


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_dataset_url_must_be_string(reader_factory):
    with pytest.raises(ValueError):
        reader_factory(None)

    with pytest.raises(ValueError):
        reader_factory(123)

    with pytest.raises(ValueError):
        reader_factory([])


def test_diagnostics_reader_v1(synthetic_dataset):
    with make_reader(synthetic_dataset.url) as reader:
        next(reader)
        diags = reader.diagnostics
        # Hard to make a meaningful assert on the content of the diags without potentially introducing a race
        assert 'output_queue_size' in diags


def test_diagnostics_reader_v2(synthetic_dataset):
    with make_reader(synthetic_dataset.url, reader_engine='experimental_reader_v2') as reader:
        next(reader)
        diags = reader.diagnostics
        # Hard to make a meaningful assert on the content of the diags without potentially introducing a race
        assert 'output_queue_size' in diags


@pytest.mark.skip('We no longer know how many rows in each row group')
def test_normalize_shuffle_partitions(synthetic_dataset):
    dataset = pq.ParquetDataset(synthetic_dataset.path)
    row_drop_partitions = Reader._normalize_shuffle_options(2, dataset)
    assert row_drop_partitions == 2

    row_drop_partitions = Reader._normalize_shuffle_options(1000, dataset)
    assert row_drop_partitions == 10


def test_bound_size_of_output_queue_size_reader(synthetic_dataset):
    """This test is timing sensitive so it might become flaky"""
    TIME_TO_GET_TO_STATIONARY_STATE = 0.5

    with make_reader(synthetic_dataset.url, reader_pool_type='process', workers_count=1) as reader:
        assert 0 == reader.diagnostics['items_produced']
        next(reader)
        # Verify that we did not consume all rowgroups (should be 10) and ventilator throttles number of ventilated
        # items
        sleep(TIME_TO_GET_TO_STATIONARY_STATE)
        assert reader.diagnostics['items_consumed'] < 5
        assert reader.diagnostics['items_inprocess'] < 5


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_invalid_cache_type(synthetic_dataset, reader_factory):
    with pytest.raises(ValueError, match='Unknown cache_type'):
        reader_factory(synthetic_dataset.url, cache_type='bogus_cache_type')


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_invalid_reader_pool_type(synthetic_dataset, reader_factory):
    with pytest.raises(ValueError, match='Unknown reader_pool_type'):
        reader_factory(synthetic_dataset.url, reader_pool_type='bogus_pool_type')


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_invalid_reader_engine(synthetic_dataset, reader_factory):
    with pytest.raises(ValueError, match='Supported reader_engine values'):
        make_reader(synthetic_dataset.url, reader_engine='bogus reader engine')


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_reader_engine_v2_with_transform_is_not_supported(synthetic_dataset, reader_factory):
    with pytest.raises(NotImplementedError):
        make_reader(synthetic_dataset.url, reader_engine='experimental_reader_v2',
                    transform_spec=TransformSpec(lambda x: x))
