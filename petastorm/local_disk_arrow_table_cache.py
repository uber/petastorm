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

import pyarrow as pa

from petastorm.local_disk_cache import LocalDiskCache


class LocalDiskArrowTableCache(LocalDiskCache):
    """A disk cache implementation """
    def __init__(self, *args, **kwargs):
        super(LocalDiskArrowTableCache, self).__init__(*args, **kwargs)
        # Workaround for https://issues.apache.org/jira/browse/ARROW-5260
        # unless we try to serialize something before deserialize_components is called, we would crash with a sigsegv
        pa.serialize(0)

    def get(self, key, fill_cache_func):
        value = self._cache.get(key, default=None)
        if value is None:
            value = fill_cache_func()
            table_pandas = value.to_pandas()
            serialized_df = pa.serialize(table_pandas)
            components = serialized_df.to_components()
            self._cache.set(key, components)
        else:
            original_df = pa.deserialize_components(value)
            value = pa.Table.from_pandas(original_df, preserve_index=False)

        return value
