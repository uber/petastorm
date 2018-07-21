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
from decimal import Decimal

import numpy as np
from pyspark.sql.types import StringType, ByteType, ShortType, IntegerType, LongType, DecimalType

from dataset_toolkit.codecs import NdarrayCodec, ScalarCodec
from dataset_toolkit.unischema import UnischemaField


class NumpyArrayCodecsTest(unittest.TestCase):

    def test_numpy_codec(self):
        SHAPE = (10, 20, 30)
        expected = np.random.rand(*SHAPE).astype(dtype=np.int32)
        codec = NdarrayCodec()
        field = UnischemaField(name='test_name', numpy_dtype=np.int32, shape=SHAPE, codec=NdarrayCodec(),
                               nullable=False)
        np.testing.assert_equal(codec.decode(field, codec.encode(field, expected)), expected)


class ScalarCodecsTest(unittest.TestCase):

    def test_scalar_codec_string(self):
        codec = ScalarCodec(StringType())
        field = UnischemaField(name='field_string', numpy_dtype=np.string_, shape=(), codec=codec, nullable=False)

        self.assertEqual(codec.decode(field, codec.encode(field, 'abc')), 'abc')
        self.assertEqual(codec.decode(field, codec.encode(field, '')), '')

    def _test_scalar_type(self, spark_type, numpy_type, bits):
        codec = ScalarCodec(spark_type())
        field = UnischemaField(name='field_int', numpy_dtype=numpy_type, shape=(), codec=codec, nullable=False)

        min_val, max_val = -2 ** (bits - 1), 2 ** (bits - 1) - 1
        self.assertEqual(codec.decode(field, codec.encode(field, numpy_type(min_val))), min_val)
        self.assertEqual(codec.decode(field, codec.encode(field, numpy_type(max_val))), max_val)
        self.assertNotEqual(codec.decode(field, codec.encode(field, numpy_type(min_val))), min_val - 1)

    def test_scalar_codec_byte(self):
        self._test_scalar_type(ByteType, np.int8, 8)
        self._test_scalar_type(ShortType, np.int16, 16)
        self._test_scalar_type(IntegerType, np.int32, 32)
        self._test_scalar_type(LongType, np.int64, 64)

    def test_scalar_codec_decimal(self):
        codec = ScalarCodec(DecimalType(4, 3))
        field = UnischemaField(name='field_decimal', numpy_dtype=Decimal, shape=(), codec=codec, nullable=False)

        value = Decimal('123.4567')
        self.assertEqual(codec.decode(field, codec.encode(field, value)), value)


if __name__ == '__main__':
    # Delegate to the test framework.
    unittest.main()
