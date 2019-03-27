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

import numpy as np
import pytest

from petastorm.codecs import NdarrayCodec
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
