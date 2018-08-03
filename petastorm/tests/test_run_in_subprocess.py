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
from functools import partial

from petastorm.unischema import dict_to_spark_row, Unischema
from petastorm.utils import run_in_subprocess


def builtin_func():
    return list(range(10))


def multiply(a, b):
    return a * b


class RunInSubprocessTest(unittest.TestCase):

    def test_run_in_subprocess(self):
        # Serialization of a built in function
        self.assertEquals(run_in_subprocess(builtin_func), builtin_func())

        # Arg passing
        self.assertEquals(run_in_subprocess(multiply, 2, 3), 6)

    def test_partial_application(self):
        unischema = Unischema('foo', [])
        func = partial(dict_to_spark_row, unischema)
        func({})

        # Must pass as positional arg in the right order
        func = partial(dict_to_spark_row, {})
        with self.assertRaises(AssertionError):
            func(Unischema)


if __name__ == '__main__':
    unittest.main()
