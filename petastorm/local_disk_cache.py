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
from __future__ import division

import pickle
import shutil
from diskcache import FanoutCache

from petastorm.cache import CacheBase


class LocalDiskCache(CacheBase):
    def __init__(self, path, size_limit_bytes, expected_row_size_bytes, shards=6, cleanup=False, **settings):
        """LocalDiskCache is an adapter to a diskcache implementation.

        LocalDiskCache can be used by a petastorm Reader class to temporarily keep parts of the dataset on a local
        file system storage.

        :param path: Path where the dataset cache is being stored.
        :param size_limit_bytes: Maximal size of the disk-space to be used by cache. The size of the cache may actually
                                 grow somewhat above the size_limit_bytes, so the limit is not very strict.
        :param expected_row_size_bytes: Approximate size of a single row. This argument is used to perform a sanity
                                 check on the capacity of individual shards.
        :param shards: Cache can be sharded. Larger number of shards improve writing parallelism.
        :param cleanup: If set to True, cache directory would be removed when cleanup() method is called.
        :param settings: these parameters passed-through to the diskcache.Cache class.
                         For details, see: http://www.grantjenks.com/docs/diskcache/tutorial.html#settings
        """
        default_settings = {
            'size_limit': size_limit_bytes,
            'eviction_policy': 'least-recently-stored',
            'disk_pickle_protocol': pickle.HIGHEST_PROTOCOL,
        }
        default_settings.update(settings)

        if default_settings['eviction_policy'] != 'none' and size_limit_bytes / shards < 5 * expected_row_size_bytes:
            raise ValueError('Condition \'size_limit_bytes / shards < 5 * expected_row_size_bytes\' needs to hold, '
                             'otherwise, newly added cached values might end up being immediately evicted.')

        self._cleanup = cleanup
        self._path = path
        self._size_limit_bytes = size_limit_bytes
        self._default_settings = default_settings
        self._cache = FanoutCache(path, shards, **default_settings)

    def get(self, key, fill_cache_func):
        value = self._cache.get(key, default=None)
        if value is None:
            value = fill_cache_func()
            # If eviction policy is set to 'none', we don't store the value if the cache is full
            if self._default_settings['eviction_policy'] == 'none':
                if self._cache.volume() < self._size_limit_bytes:
                    self._cache.set(key, value)
            else:  # evict and store the value
                self._cache.set(key, value)
        return value

    def cleanup(self):
        if self._cleanup:
            try:
                # Very important to stop background threads
                self._cache.close()
            except Exception as e:
                print(f"Error closing cache: {e}", flush=True)
            try:
                shutil.rmtree(self._path, ignore_errors=False)
            except FileNotFoundError:
                # OK: it's already gone
                pass
            except Exception as e:
                print(f"Error during rmtree: {e}", flush=True)
