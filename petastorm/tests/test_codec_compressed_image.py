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
import io

import numpy as np
import pytest
from PIL import Image

from petastorm.codecs import CompressedImageCodec
from petastorm.unischema import UnischemaField


def test_png():
    """Simple noop encode/decode using png codec. Verify that supports uint16 png codec and monochrome and
    color images."""
    for size in [(300, 200), (300, 200, 3)]:
        for dtype in [np.uint8, np.uint16]:
            expected_image = np.random.randint(0, np.iinfo(dtype).max, size=size, dtype=dtype)
            codec = CompressedImageCodec('png')
            field = UnischemaField(name='field_image', numpy_dtype=dtype, shape=size, codec=codec,
                                   nullable=False)

            actual_image = codec.decode(field, codec.encode(field, expected_image))
            np.testing.assert_array_equal(expected_image, actual_image)
            assert expected_image.dtype == actual_image.dtype


def test_jpeg():
    """Test lossy image codec"""
    for size in [(300, 200), (300, 200, 3)]:
        expected_image = np.random.randint(0, 255, size=size, dtype=np.uint8)
        codec = CompressedImageCodec('jpeg', quality=100)
        field = UnischemaField(name='field_image', numpy_dtype=np.uint8, shape=size, codec=codec, nullable=False)

        actual_image = codec.decode(field, codec.encode(field, expected_image))
        # Check a non exact match between the images. Verifying reasonable mean absolute error (up to 10)
        mean_abs_error = np.mean(np.abs(expected_image.astype(np.float) - actual_image.astype(np.float)))
        # The threshold is relatively high as compressing random images with jpeg results in a significant
        # quality loss
        assert mean_abs_error < 50
        assert np.any(expected_image != actual_image, axis=None)


def test_jpeg_quality():
    """Compare mean abs error between different encoding quality settings. Higher quality value should result
    in a smaller error"""
    size = (300, 200, 3)
    expected_image = np.random.randint(0, 255, size=size, dtype=np.uint8)

    errors = dict()
    for quality in [10, 99]:
        codec = CompressedImageCodec('jpeg', quality=quality)
        field = UnischemaField(name='field_image', numpy_dtype=np.uint8, shape=size, codec=codec, nullable=False)
        actual_image = codec.decode(field, codec.encode(field, expected_image))
        errors[quality] = np.mean(np.abs(expected_image.astype(np.float) - actual_image.astype(np.float)))

    assert errors[10] > errors[99]


def test_bad_shape():
    codec = CompressedImageCodec('png')
    field = UnischemaField(name='field_image', numpy_dtype=np.uint8, shape=(10, 20), codec=codec, nullable=False)
    with pytest.raises(ValueError, match='Unexpected dimensions'):
        codec.encode(field, np.zeros((100, 200), dtype=np.uint8))


def test_bad_dtype():
    codec = CompressedImageCodec('png')
    field = UnischemaField(name='field_image', numpy_dtype=np.uint8, shape=(10, 20), codec=codec, nullable=False)
    with pytest.raises(ValueError, match='Unexpected type'):
        codec.encode(field, np.zeros((100, 200), dtype=np.uint16))


def test_cross_coding():
    """Encode using PIL and decode using opencv. Previously had an error with channel ordering. This test
    covers this issue for the future """
    for size in [(300, 200), (300, 200, 3)]:
        dtype = np.uint8
        expected_image = np.random.randint(0, np.iinfo(dtype).max, size=size, dtype=np.uint8)
        codec = CompressedImageCodec('png')
        field = UnischemaField(name='field_image', numpy_dtype=dtype, shape=size, codec=codec,
                               nullable=False)

        encoded = Image.fromarray(expected_image)
        encoded_bytes = io.BytesIO()
        encoded.save(encoded_bytes, format='PNG')

        actual_image = codec.decode(field, encoded_bytes.getvalue())
        np.testing.assert_array_equal(expected_image, actual_image)
        assert expected_image.dtype == actual_image.dtype


def test_invalid_image_size():
    """Codec can encode only (H, W) and (H, W, 3) images"""
    codec = CompressedImageCodec('png')

    field = UnischemaField(name='field_image', numpy_dtype=np.uint8, shape=(10, 10, 3), codec=codec,
                           nullable=False)

    with pytest.raises(ValueError):
        codec.encode(field, np.zeros((10,), dtype=np.uint8))

    with pytest.raises(ValueError):
        codec.encode(field, np.zeros((10, 10, 2), dtype=np.uint8))

    with pytest.raises(ValueError):
        codec.encode(field, np.zeros((10, 10, 10, 10), dtype=np.uint8))
