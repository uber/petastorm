# -*- coding: utf-8 -*-

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
import pickle
from decimal import Decimal

import numpy as np

from petastorm.reader_impl.pyarrow_serializer import PyArrowSerializer


def test_nominal():
    s = PyArrowSerializer()
    expected = [{'a': np.asarray([1, 2], dtype=np.uint64)}]
    actual = s.deserialize(s.serialize(expected))
    np.testing.assert_array_equal(actual[0]['a'], expected[0]['a'])


def test_serializer_is_pickable():
    """Pickle/depickle the serializer to make sure it can be passed
    as a parameter cross process boundaries when using futures"""
    s = PyArrowSerializer()
    deserialized_s = pickle.loads(pickle.dumps(s))

    expected = [{'a': np.asarray([1, 2], dtype=np.uint64)}]
    actual = deserialized_s.deserialize(deserialized_s.serialize(expected))
    np.testing.assert_array_equal(actual[0]['a'], expected[0]['a'])


def test_decimal():
    s = PyArrowSerializer()
    expected = [{'a': Decimal('1.2')}]
    actual = s.deserialize(s.serialize(expected))
    np.testing.assert_array_equal(actual[0]['a'], expected[0]['a'])

    expected = [{'a': [Decimal('1.2')]}]
    actual = s.deserialize(s.serialize(expected))
    np.testing.assert_array_equal(actual[0]['a'], expected[0]['a'])


def test_all_matrix_types():
    s = PyArrowSerializer()
    # We would be using serializer with arrays of dictionaries or arrays of dictionaries of dictionaries (ngram)
    serialized_values = [
        (np.int8, -127),
        (np.uint8, 255),
        (np.int16, -2 ** 15),
        (np.uint16, 2 ** 16 - 1),
        (np.int32, -2 ** 31),
        (np.uint32, 2 ** 32 - 1),
        (np.float16, 1.2),
        (np.float32, 1.2),
        (np.float64, 1.2),
        (np.string_, 'abc'),
        (np.unicode_, u'אבג'),
        (np.int64, -2 ** 63),
        (np.uint64, 2 ** 64 - 1),
    ]

    for type_factory, value in serialized_values:
        desired = [{'value': np.asarray(4 * [value], dtype=type_factory)}]
        actual = s.deserialize(s.serialize(desired))
        assert actual[0]['value'].dtype == desired[0]['value'].dtype
        np.testing.assert_array_equal(actual[0]['value'], desired[0]['value'])
