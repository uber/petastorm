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
import pytest
from pyspark.sql.types import DoubleType

from petastorm.codecs import NdarrayCodec, ScalarCodec
from petastorm.unischema import UnischemaField, Unischema
from petastorm.utils import decode_row, DecodeFieldError

MatrixField = UnischemaField('matrix', np.float64, (10, 10), NdarrayCodec(), False)
MatrixSchema = Unischema('TestSchema', [MatrixField])


def test_nominal_case():
    """Nominal flow: can decode field successfully"""
    expected = np.random.rand(10, 10)
    row = {'matrix': NdarrayCodec().encode(MatrixField, expected)}

    actual = decode_row(row, MatrixSchema)['matrix']

    np.testing.assert_equal(actual, expected)


def test_can_not_decode():
    """Make sure field name is part of the error message"""
    row = {'matrix': 'bogus'}

    with pytest.raises(DecodeFieldError, match='matrix'):
        decode_row(row, MatrixSchema)


def test_decode_numpy_scalar_when_codec_is_none():
    """Decoding a row that has a field with the codec set to None. The type should be deduced automatically
    from UnischemaField's numpy_dtype attribute"""

    MatrixSchema = Unischema('TestSchema', [UnischemaField('scalar', np.float64, ())])
    row = {'scalar': 42.0}
    decoded_value = decode_row(row, MatrixSchema)['scalar']
    assert decoded_value == 42
    assert isinstance(decoded_value, np.float64)


def test_decode_decimal_scalar_when_codec_is_none():
    """Decoding a row that has a field with the codec set to None. The type should be deduced automatically
    from UnischemaField's numpy_dtype attribute if the type is either a numpy scalar or a Decimal"""

    MatrixSchema = Unischema('TestSchema', [UnischemaField('scalar', Decimal, ())])

    row = {'scalar': '123.45'}
    decoded_value = decode_row(row, MatrixSchema)['scalar']
    assert decoded_value == Decimal('123.45')
    assert isinstance(decoded_value, Decimal)

    row = {'scalar': Decimal('123.45')}
    decoded_value = decode_row(row, MatrixSchema)['scalar']
    assert decoded_value == Decimal('123.45')
    assert isinstance(decoded_value, Decimal)


def test_decode_numpy_scalar_with_explicit_scalar_codec():
    """Decoding a row that has a field with the codec set explicitly"""

    MatrixSchema = Unischema('TestSchema', [UnischemaField('scalar', np.float64, (), ScalarCodec(DoubleType()), False)])
    row = {'scalar': 42.0}
    decoded_value = decode_row(row, MatrixSchema)['scalar']
    assert decoded_value == 42
    assert isinstance(decoded_value, np.float64)


def test_decode_numpy_scalar_with_unknown_dtype():
    """If numpy_dtype is None, then the value is not decoded, just passed through."""

    MatrixSchema = Unischema('TestSchema', [UnischemaField('scalar', None, ())])
    row = {'scalar': [4, 2]}
    decoded_value = decode_row(row, MatrixSchema)['scalar']
    assert decoded_value == [4, 2]
