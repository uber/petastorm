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

import abc

import six


@six.add_metaclass(abc.ABCMeta)
class CacheBase(object):
    @abc.abstractmethod
    def get(self, key, fill_cache_func):
        """Gets an entry from the cache implementation.

        If there is a cache miss, ``fill_cache_func()`` will be evaluated to get the value.

        :param key: A key identifying cache entry
        :param fill_cache_func: This function will be evaluated (``fill_cache_func()``) to populate cache, if no
            value is present in the cache.
        :return: A value from cache
        """
        pass


class NullCache(CacheBase):
    """A pass-through cache implementation: value generating function will be called each."""

    def get(self, key, fill_cache_func):
        return fill_cache_func()
