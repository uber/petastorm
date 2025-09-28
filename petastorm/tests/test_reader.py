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

from petastorm import make_reader
from petastorm.reader import Reader

# pylint: disable=unnecessary-lambda
READER_FACTORIES = [
    make_reader,
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


def test_disable_zmq_copy_buffers_in_reader(synthetic_dataset):
    """Assert that the underlying ProcessPool has zmq_copy_buffers disabled"""

    with make_reader(synthetic_dataset.url, reader_pool_type='process',
                     workers_count=1, zmq_copy_buffers=False) as reader:
        assert not reader.diagnostics['zmq_copy_buffers']


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_invalid_cache_type(synthetic_dataset, reader_factory):
    with pytest.raises(ValueError, match='Unknown cache_type'):
        reader_factory(synthetic_dataset.url, cache_type='bogus_cache_type')


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_invalid_reader_pool_type(synthetic_dataset, reader_factory):
    with pytest.raises(ValueError, match='Unknown reader_pool_type'):
        reader_factory(synthetic_dataset.url, reader_pool_type='bogus_pool_type')


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_deprecated_shard_seed(synthetic_dataset, reader_factory):
    match_str = 'shard_seed was deprecated and will be removed in future versions.'
    with pytest.warns(UserWarning, match=match_str):
        reader_factory(synthetic_dataset.url, shard_seed=123)


def test_cleanup_cache_with_local_disk_cache(synthetic_dataset, tmpdir):
    """Test that cleanup_cache properly removes local disk cache directory"""
    import os
    cache_location = tmpdir.strpath

    with make_reader(synthetic_dataset.url,
                     cache_type='local-disk',
                     cache_location=cache_location,
                     cache_size_limit=1000000,
                     cache_row_size_estimate=100) as reader:
        # Read some data to populate cache
        next(reader)
        # Cache directory should exist
        assert os.path.exists(cache_location)

    # After context manager exit, cache should be cleaned up
    assert not os.path.exists(cache_location)


def test_cleanup_cache_with_null_cache(synthetic_dataset, capsys):
    """Test that cleanup_cache works properly with null cache"""
    with make_reader(synthetic_dataset.url, cache_type='null') as reader:
        next(reader)
        # Manually call cleanup_cache to test it
        reader.cleanup_cache()

    # Check that cleanup message was printed
    captured = capsys.readouterr()
    assert "Cache cleanup complete." in captured.out


def test_cleanup_cache_manual_call(synthetic_dataset, tmpdir):
    """Test manually calling cleanup_cache method"""
    import os
    cache_location = tmpdir.strpath

    reader = make_reader(synthetic_dataset.url,
                        cache_type='local-disk',
                        cache_location=cache_location,
                        cache_size_limit=1000000,
                        cache_row_size_estimate=100)

    try:
        next(reader)
        assert os.path.exists(cache_location)

        # Manually call cleanup_cache
        reader.cleanup_cache()
        assert not os.path.exists(cache_location)
    finally:
        reader.stop()
        reader.join()


def test_cleanup_cache_exception_handling(synthetic_dataset, tmpdir, capsys, monkeypatch):
    """Test that cleanup_cache handles exceptions gracefully"""
    import os
    from petastorm.local_disk_cache import LocalDiskCache

    cache_location = tmpdir.strpath

    with make_reader(synthetic_dataset.url,
                     cache_type='local-disk',
                     cache_location=cache_location,
                     cache_size_limit=1000000,
                     cache_row_size_estimate=100) as reader:
        next(reader)

        # Mock the cleanup method to raise an exception
        def mock_cleanup():
            raise RuntimeError("Simulated cleanup error")

        monkeypatch.setattr(reader.cache, 'cleanup', mock_cleanup)

        # Call cleanup_cache - should handle exception gracefully
        reader.cleanup_cache()

        # Check that error message was printed
        captured = capsys.readouterr()
        assert "Error cleaning cache: Simulated cleanup error" in captured.out
        assert "Cache cleanup complete." in captured.out
