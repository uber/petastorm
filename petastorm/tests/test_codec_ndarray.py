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

from petastorm.codecs import NdarrayCodec, CompressedNdarrayCodec
from petastorm.unischema import UnischemaField

NUMERIC_DTYPES = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float32,
                  np.float64]


@pytest.mark.parametrize('codec_factory', [
    NdarrayCodec,
    CompressedNdarrayCodec,
])
def test_ndarray_codec(codec_factory):
    SHAPE = (10, 20, 3)
    for dtype in NUMERIC_DTYPES:
        expected = np.random.rand(*SHAPE).astype(dtype=dtype)
        codec = codec_factory()
        field = UnischemaField(name='test_name', numpy_dtype=dtype, shape=SHAPE, codec=codec, nullable=False)
        actual = codec.decode(field, codec.encode(field, expected))
        np.testing.assert_equal(actual, expected)
        assert expected.dtype == actual.dtype
