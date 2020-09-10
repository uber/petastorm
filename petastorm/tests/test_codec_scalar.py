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
from pyspark.sql.types import StringType, ByteType, ShortType, IntegerType, LongType, DecimalType, BooleanType, \
    FloatType

from petastorm.codecs import ScalarCodec
from petastorm.unischema import UnischemaField


def test_byte_string():
    codec = ScalarCodec(StringType())
    field = UnischemaField(name='field_string', numpy_dtype=np.string_, shape=(), codec=codec, nullable=False)

    assert codec.decode(field, codec.encode(field, 'abc')) == b'abc'
    assert codec.decode(field, codec.encode(field, '')) == b''


def test_unicode():
    codec = ScalarCodec(StringType())
    field = UnischemaField(name='field_string', numpy_dtype=np.unicode_, shape=(), codec=codec, nullable=False)

    assert codec.decode(field, codec.encode(field, 'abc')) == 'abc'
    assert codec.decode(field, codec.encode(field, '')) == ''


@pytest.mark.parametrize('spark_numpy_types', [
    (ByteType, np.uint8),
    (ByteType, np.int8),
    (ShortType, np.int16),
    (IntegerType, np.int32),
    (LongType, np.int64),
])
def test_numeric_types(spark_numpy_types):
    spark_type, numpy_type = spark_numpy_types

    codec = ScalarCodec(spark_type())
    field = UnischemaField(name='field_int', numpy_dtype=numpy_type, shape=(), codec=codec, nullable=False)

    min_val, max_val = np.iinfo(numpy_type).min, np.iinfo(numpy_type).max

    assert codec.decode(field, codec.encode(field, numpy_type(min_val))) == min_val
    assert codec.decode(field, codec.encode(field, numpy_type(max_val))) == max_val


def test_scalar_codec_decimal():
    codec = ScalarCodec(DecimalType(4, 3))
    field = UnischemaField(name='field_decimal', numpy_dtype=Decimal, shape=(), codec=codec, nullable=False)

    value = Decimal('123.4567')
    assert codec.decode(field, codec.encode(field, value)) == value


def test_bad_encoded_data_shape():
    codec = ScalarCodec(IntegerType())
    field = UnischemaField(name='field_int', numpy_dtype=np.int32, shape=(), codec=codec, nullable=False)
    with pytest.raises(TypeError):
        codec.decode(field, codec.encode(field, np.asarray([10, 10])))


def test_bad_unischema_field_shape():
    codec = ScalarCodec(IntegerType())
    field = UnischemaField(name='field_int', numpy_dtype=np.int32, shape=(1,), codec=codec, nullable=False)
    with pytest.raises(ValueError, match='must be an empty tuple'):
        codec.encode(field, np.int32(1))


def test_encode_scalar_bool():
    codec = ScalarCodec(BooleanType())
    field = UnischemaField(name='field_bool', numpy_dtype=np.bool, shape=(), codec=codec, nullable=False)

    encoded = codec.encode(field, np.bool_(True))
    assert isinstance(codec.encode(field, encoded), bool)
    assert encoded

    encoded = codec.encode(field, np.bool_(False))
    assert not encoded


def test_encode_scalar_int():
    codec = ScalarCodec(IntegerType())
    field = UnischemaField(name='field_int', numpy_dtype=np.int32, shape=(), codec=codec, nullable=False)
    encoded = codec.encode(field, np.int32(42))
    assert isinstance(encoded, int)
    assert 42 == encoded


def test_encode_scalar_float():
    codec = ScalarCodec(FloatType())
    expected = np.random.random(()).astype(np.float64)
    field = UnischemaField(name='field_float', numpy_dtype=np.float32, shape=(), codec=codec, nullable=False)
    encoded = codec.encode(field, expected)
    assert isinstance(encoded, float)
    assert expected == encoded


def test_encode_scalar_string():
    codec = ScalarCodec(StringType())
    expected = 'surprise'
    field = UnischemaField(name='field_string', numpy_dtype=np.unicode_, shape=(), codec=codec, nullable=False)
    encoded = codec.encode(field, expected)
    assert isinstance(encoded, str)
    assert expected == encoded


@pytest.mark.parametrize("non_scalar_value", [[1.2], np.asarray([3.4]), [5, 6]])
def test_encode_non_scalar_type_is_passed(non_scalar_value):
    codec = ScalarCodec(FloatType())
    field = UnischemaField(name='field_float', numpy_dtype=np.float32, shape=(), codec=codec, nullable=False)
    with pytest.raises(TypeError, match='Expected a scalar'):
        codec.encode(field, non_scalar_value)


def test_str_special_method():
    codec = ScalarCodec(FloatType())
    assert str(codec) == 'ScalarCodec(FloatType())'
