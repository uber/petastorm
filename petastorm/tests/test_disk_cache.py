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

import numpy as np
import pandas as pd
import pyarrow as pa

from petastorm.local_disk_arrow_table_cache import LocalDiskArrowTableCache
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
    assert _recursive_folder_size(tmpdir.strpath) < 2 * MB


def _should_never_be_called():
    assert False, 'Should not be called'


def test_arrow_table_caching(tmpdir):
    cache = LocalDiskArrowTableCache(tmpdir.strpath, 10 * MB, 4)

    df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))
    dummy_table = pa.Table.from_pandas(df)

    table_from_cache = cache.get('my_key', lambda: dummy_table)
    assert table_from_cache == dummy_table

    cache.get('my_key', _should_never_be_called)
