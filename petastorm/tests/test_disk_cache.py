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
import shutil
from unittest.mock import patch

import numpy as np
import pytest

from petastorm.local_disk_cache import LocalDiskCache

MB = 2 ** 20
KB = 2 ** 10


def _recursive_folder_size(folder):
    folder_size = 0
    for (path, _, files) in os.walk(folder):
        for flename in files:
            filename = os.path.join(path, flename)
            folder_size += os.path.getsize(filename)
    return folder_size


def test_simple_scalar_cache(tmpdir):
    """Testing trivial NullCache: should trigger value generating function on each run"""
    cache = LocalDiskCache(tmpdir.strpath, 1 * MB, 4)
    assert 42 == cache.get('some_key', lambda: 42)
    assert 42 == cache.get('some_key', lambda: 43)


def test_size_limit_constraint(tmpdir):
    """Testing trivial NullCache: should trigger value generating function on each run"""
    # We will write total of 5MB to the cache (50KB items x 100)
    RECORD_SIZE_BYTES = 50 * KB
    RECORDS_COUNT = 100

    a_record = np.random.randint(0, 255, (RECORD_SIZE_BYTES,), np.uint8)
    cache = LocalDiskCache(tmpdir.strpath, 1 * MB, RECORD_SIZE_BYTES, shards=1)

    for i in range(RECORDS_COUNT):
        cache.get('some_key_{}'.format(i), lambda: a_record)

    # Check that we are more or less within the size limit
    assert _recursive_folder_size(tmpdir.strpath) < 3 * MB


def test_eviction_policy_none_large_data(tmpdir):
    """Test that the cache size limit is respected when the data is larger than the limit"""
    # We will write total of 5MB to the cache (50KB items x 100)
    RECORD_SIZE_BYTES = 100 * KB
    RECORDS_COUNT = 1000
    settings = {
        'eviction_policy': 'none'
    }

    a_record = np.random.randint(0, 255, (RECORD_SIZE_BYTES,), np.uint8)
    cache = LocalDiskCache(tmpdir.strpath, 3000 * KB, RECORD_SIZE_BYTES, shards=1, **settings)

    for i in range(RECORDS_COUNT):
        cache.get('some_key_{}'.format(i), lambda: a_record)

    # Check that we are within the size limit + overhead and way less than 10MB data.
    # Slightly more than the limit because _recursive_folder_size is actual disk space
    # and not cache size volume
    assert _recursive_folder_size(tmpdir.strpath) < 4000 * KB


def test_cleanup_with_cleanup_true(tmpdir):
    """Test that cleanup removes cache directory when cleanup=True"""
    cache_path = tmpdir.strpath
    cache = LocalDiskCache(cache_path, 1 * MB, 4, cleanup=True)

    # Add some data to the cache to ensure directory exists and has content
    cache.get('test_key', lambda: 'test_value')

    # Verify cache directory exists and has content
    assert os.path.exists(cache_path)
    assert len(os.listdir(cache_path)) > 0

    # Call cleanup
    cache.cleanup()

    # Verify cache directory is removed
    assert not os.path.exists(cache_path)


def test_cleanup_with_cleanup_false(tmpdir):
    """Test that cleanup does not remove cache directory when cleanup=False"""
    cache_path = tmpdir.strpath
    cache = LocalDiskCache(cache_path, 1 * MB, 4, cleanup=False)

    # Add some data to the cache
    cache.get('test_key', lambda: 'test_value')

    # Verify cache directory exists and has content
    assert os.path.exists(cache_path)
    assert len(os.listdir(cache_path)) > 0

    # Call cleanup
    cache.cleanup()

    # Verify cache directory still exists (cleanup=False should not remove it)
    assert os.path.exists(cache_path)


def test_cleanup_handles_cache_close_error(tmpdir):
    """Test that cleanup handles OSError when closing cache"""
    cache_path = tmpdir.strpath
    cache = LocalDiskCache(cache_path, 1 * MB, 4, cleanup=True)

    # Add some data to the cache
    cache.get('test_key', lambda: 'test_value')

    # Mock the cache.close() method to raise OSError
    with patch.object(cache._cache, 'close', side_effect=OSError("Mock close error")):
        with patch('builtins.print') as mock_print:
            # Call cleanup - should handle the error gracefully
            cache.cleanup()

            # Verify error was printed
            mock_print.assert_called_with("Error closing cache: Mock close error", flush=True)

    # Directory should still be removed despite close error
    assert not os.path.exists(cache_path)


def test_cleanup_handles_attribute_error_on_close(tmpdir):
    """Test that cleanup handles AttributeError when closing cache"""
    cache_path = tmpdir.strpath
    cache = LocalDiskCache(cache_path, 1 * MB, 4, cleanup=True)

    # Add some data to the cache
    cache.get('test_key', lambda: 'test_value')

    # Mock the cache.close() method to raise AttributeError
    with patch.object(cache._cache, 'close', side_effect=AttributeError("Mock attribute error")):
        with patch('builtins.print') as mock_print:
            # Call cleanup - should handle the error gracefully
            cache.cleanup()

            # Verify error was printed
            mock_print.assert_called_with("Error closing cache: Mock attribute error", flush=True)

    # Directory should still be removed despite close error
    assert not os.path.exists(cache_path)


def test_cleanup_handles_file_not_found_error(tmpdir):
    """Test that cleanup handles FileNotFoundError when removing directory"""
    cache_path = tmpdir.strpath
    cache = LocalDiskCache(cache_path, 1 * MB, 4, cleanup=True)

    # Add some data to the cache
    cache.get('test_key', lambda: 'test_value')

    # Manually remove the directory to simulate FileNotFoundError
    shutil.rmtree(cache_path)

    # Call cleanup - should handle FileNotFoundError gracefully
    cache.cleanup()  # Should not raise an exception


def test_cleanup_handles_rmtree_os_error(tmpdir):
    """Test that cleanup handles OSError when removing directory"""
    cache_path = tmpdir.strpath
    cache = LocalDiskCache(cache_path, 1 * MB, 4, cleanup=True)

    # Add some data to the cache
    cache.get('test_key', lambda: 'test_value')

    # Mock shutil.rmtree to raise OSError
    with patch('shutil.rmtree', side_effect=OSError("Mock rmtree error")):
        with patch('builtins.print') as mock_print:
            # Call cleanup - should handle the error gracefully
            cache.cleanup()

            # Verify error was printed
            mock_print.assert_called_with("Error during rmtree: Mock rmtree error", flush=True)


def test_cleanup_multiple_calls(tmpdir):
    """Test that cleanup can be called multiple times safely"""
    cache_path = tmpdir.strpath
    cache = LocalDiskCache(cache_path, 1 * MB, 4, cleanup=True)

    # Add some data to the cache
    cache.get('test_key', lambda: 'test_value')

    # Verify cache directory exists
    assert os.path.exists(cache_path)

    # Call cleanup first time
    cache.cleanup()
    assert not os.path.exists(cache_path)

    # Call cleanup second time - should handle gracefully
    cache.cleanup()  # Should not raise an exception


def test_size_limit_validation_error(tmpdir):
    """Test that ValueError is raised when size_limit_bytes / shards < 5 * expected_row_size_bytes"""
    cache_path = tmpdir.strpath
    size_limit_bytes = 1000  # 1KB
    expected_row_size_bytes = 500  # 500 bytes
    shards = 1

    # This should violate the condition: 1000 / 1 = 1000 < 5 * 500 = 2500
    expected_error = ("Condition 'size_limit_bytes / shards < 5 \\* expected_row_size_bytes' "
                      "needs to hold")
    with pytest.raises(ValueError, match=expected_error):
        LocalDiskCache(cache_path, size_limit_bytes, expected_row_size_bytes, shards=shards)


def test_size_limit_validation_passes(tmpdir):
    """Test that no error is raised when size_limit_bytes / shards >= 5 * expected_row_size_bytes"""
    cache_path = tmpdir.strpath
    size_limit_bytes = 10000  # 10KB
    expected_row_size_bytes = 500  # 500 bytes
    shards = 1

    # This should satisfy the condition: 10000 / 1 = 10000 >= 5 * 500 = 2500
    cache = LocalDiskCache(cache_path, size_limit_bytes, expected_row_size_bytes, shards=shards)

    # Should be able to use the cache normally
    assert cache.get('test_key', lambda: 'test_value') == 'test_value'


def test_size_limit_validation_with_multiple_shards(tmpdir):
    """Test size limit validation with multiple shards"""
    cache_path = tmpdir.strpath
    size_limit_bytes = 5000  # 5KB
    expected_row_size_bytes = 200  # 200 bytes
    shards = 10

    # This should violate the condition: 5000 / 10 = 500 < 5 * 200 = 1000
    expected_error = ("Condition 'size_limit_bytes / shards < 5 \\* expected_row_size_bytes' "
                      "needs to hold")
    with pytest.raises(ValueError, match=expected_error):
        LocalDiskCache(cache_path, size_limit_bytes, expected_row_size_bytes, shards=shards)


def test_size_limit_validation_bypassed_with_eviction_none(tmpdir):
    """Test that size limit validation is bypassed when eviction_policy is 'none'"""
    cache_path = tmpdir.strpath
    size_limit_bytes = 1000  # 1KB
    expected_row_size_bytes = 500  # 500 bytes
    shards = 1
    settings = {'eviction_policy': 'none'}

    # This would normally violate the condition, but should pass with eviction_policy='none'
    cache = LocalDiskCache(cache_path, size_limit_bytes, expected_row_size_bytes,
                           shards=shards, **settings)

    # Should be able to use the cache normally
    assert cache.get('test_key', lambda: 'test_value') == 'test_value'


def _should_never_be_called():
    assert False, 'Should not be called'
