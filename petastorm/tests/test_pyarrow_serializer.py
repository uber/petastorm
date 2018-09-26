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

from decimal import Decimal

import numpy as np

from petastorm.reader_impl.pyarrow_serializer import PyArrowSerializer


def test_nominal():
    s = PyArrowSerializer()
    # We would be using serializer with arrays of dictionaries or arrays of dictionaries of dictionaries (ngram)
    expected = [{'a': np.asarray([1, 2], dtype=np.uint64)}]
    actual = s.deserialize(s.serialize(expected))
    np.testing.assert_array_equal(actual[0]['a'], expected[0]['a'])


def test_decimal():
    s = PyArrowSerializer()
    # We would be using serializer with arrays of dictionaries or arrays of dictionaries of dictionaries (ngram)
    expected = [{'a': Decimal('1.2')}]
    actual = s.deserialize(s.serialize(expected))
    np.testing.assert_array_equal(actual[0]['a'], str(expected[0]['a']))
