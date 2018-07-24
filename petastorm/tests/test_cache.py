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

import unittest

from petastorm.cache import NullCache


class TestNullCache(unittest.TestCase):

    def test_null_cache(self):
        """Testing trivial NullCache: should trigger value generating function on each run"""
        cache = NullCache()
        self.assertEqual(42, cache.get('some_key', lambda: 42))


if __name__ == '__main__':
    # Delegate to the test framework.
    unittest.main()
