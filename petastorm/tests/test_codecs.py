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

from petastorm.codecs import NdarrayCodec, ScalarCodec, CompressedImageCodec
from petastorm.unischema import UnischemaField


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


class CompressedImageCodecsTest(unittest.TestCase):

    def test_png(self):
        """Simple noop encode/decode using png codec. Verify that supports uint16 png codec and monochrome and
        color images."""
        for size in [(300, 200), (300, 200, 3)]:
            for dtype in [np.uint8, np.uint16]:
                expected_image = np.random.randint(0, np.iinfo(dtype).max, size=size, dtype=dtype)
                codec = CompressedImageCodec('png')
                field = UnischemaField(name='field_image', numpy_dtype=dtype, shape=(), codec=codec, nullable=False)

                actual_image = codec.decode(field, codec.encode(field, expected_image))
                np.testing.assert_array_equal(expected_image, actual_image)
                self.assertEqual(expected_image.dtype, actual_image.dtype)

    def test_jpeg(self):
        """Test lossy image codec"""
        expected_image = np.random.randint(0, 255, size=(300, 200), dtype=np.uint8)
        codec = CompressedImageCodec('jpeg')
        field = UnischemaField(name='field_image', numpy_dtype=np.uint8, shape=(), codec=codec, nullable=False)

        actual_image = codec.decode(field, codec.encode(field, expected_image))
        # Check a non exact match between the images. Verifying reasonable mean absolute error (up to 10)
        mean_abs_error = np.mean(np.abs(expected_image.astype(np.float) - actual_image.astype(np.float)))
        self.assertLess(mean_abs_error, 10)
        self.assertTrue(np.any(expected_image != actual_image, axis=None))


if __name__ == '__main__':
    # Delegate to the test framework.
    unittest.main()
