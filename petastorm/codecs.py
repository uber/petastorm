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

"""
A set of dataframe-column-codecs complements the limited data type variety of spark-/pyarrow-supported datatypes,
enabling storage of numpy multidimensional arrays, as well as compressed images, into spark dataframes, and
transitively to parquet files.

NOTE: Due to the way unischema is stored alongside dataset (with pickling), changing any of these codecs class names
and fields can result in reader breakages.
"""
from abc import abstractmethod
from io import BytesIO

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

import numpy as np
from pyspark.sql.types import BinaryType, LongType, IntegerType, ShortType, ByteType, StringType, \
    FloatType, DoubleType, BooleanType


class DataframeColumnCodec(object):
    """The abstract base class of codecs."""

    @abstractmethod
    def encode(self, unischema_field, value):
        raise RuntimeError('Abstract method was called')

    @abstractmethod
    def decode(self, unischema_field, value):
        raise RuntimeError('Abstract method was called')

    @abstractmethod
    def spark_dtype(self):
        """Spark datatype to be used for underlying storage"""
        raise RuntimeError('Abstract method was called')


class CompressedImageCodec(DataframeColumnCodec):
    def __init__(self, image_codec='png', quality=80):
        """CompressedImageCodec would compress/encompress images.

        :param image_codec: any format string supported by opencv. e.g. ``png``, ``jpeg``
        :param quality: used when using ``jpeg`` lossy compression
        """
        assert OPENCV_AVAILABLE, "{} requires opencv-python to be installed".format(type(self).__name__)

        self._image_codec = '.' + image_codec
        self._quality = quality

    def encode(self, unischema_field, value):
        """Encodes the image using OpenCV."""
        if unischema_field.numpy_dtype != value.dtype:
            raise ValueError("Unexpected type of {} feature, expected {}, got {}".format(
                unischema_field.name, unischema_field.numpy_dtype, value.dtype
            ))

        if not _is_compliant_shape(value.shape, unischema_field.shape):
            raise ValueError("Unexpected dimensions of {} feature, expected {}, got {}".format(
                unischema_field.name, unischema_field.shape, value.shape
            ))

        if len(value.shape) == 2:
            # Greyscale image
            image_bgr_or_gray = value
        elif len(value.shape) == 3 and value.shape[2] == 3:
            # Convert RGB to BGR
            image_bgr_or_gray = value[:, :, (2, 1, 0)]
        else:
            raise ValueError('Unexpected image dimensions. Supported dimensions are (H, W) or (H, W, 3). '
                             'Got {}'.format(value.shape))

        _, contents = cv2.imencode(self._image_codec,
                                   image_bgr_or_gray,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        return bytearray(contents)

    def decode(self, unischema_field, value):
        """Decodes the image using OpenCV."""

        # cv returns a BGR or grayscale image. Convert to RGB (unless a grayscale image).
        image_bgr_or_gray = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if len(image_bgr_or_gray.shape) == 2:
            # Greyscale image
            return image_bgr_or_gray
        elif len(image_bgr_or_gray.shape) == 3 and image_bgr_or_gray.shape[2] == 3:
            # Convert BGR to RGB (opencv assumes BGR)
            image_rgb = image_bgr_or_gray[:, :, (2, 1, 0)]
            return image_rgb
        else:
            raise ValueError('Unexpected image dimensions. Supported dimensions are (H, W) or (H, W, 3). '
                             'Got {}'.format(image_bgr_or_gray.shape))

    def spark_dtype(self):
        return BinaryType()


class NdarrayCodec(DataframeColumnCodec):
    """Encodes numpy ndarray into, or decodes an ndarray from, a spark dataframe field."""

    def encode(self, unischema_field, value):
        expected_dtype = unischema_field.numpy_dtype
        if isinstance(value, np.ndarray):
            if expected_dtype != value.dtype.type:
                raise ValueError('Unexpected type of {} feature. '
                                 'Expected {}. Got {}'.format(unischema_field.name, expected_dtype, value.dtype))

            expected_shape = unischema_field.shape
            if not _is_compliant_shape(value.shape, expected_shape):
                raise ValueError('Unexpected dimensions of {} feature. '
                                 'Expected {}. Got {}'.format(unischema_field.name, expected_shape, value.shape))
        else:
            raise ValueError('Unexpected type of {} feature. '
                             'Expected ndarray of {}. Got {}'.format(unischema_field.name, expected_dtype, type(value)))

        memfile = BytesIO()
        np.save(memfile, value)
        return bytearray(memfile.getvalue())

    def decode(self, unischema_field, value):
        memfile = BytesIO(value)
        return np.load(memfile)

    def spark_dtype(self):
        return BinaryType()


class CompressedNdarrayCodec(DataframeColumnCodec):
    """Encodes numpy ndarray with compression into a spark dataframe field"""

    def encode(self, unischema_field, value):
        expected_dtype = unischema_field.numpy_dtype
        if isinstance(value, np.ndarray):
            if expected_dtype != value.dtype.type:
                raise ValueError('Unexpected type of {} feature. '
                                 'Expected {}. Got {}'.format(unischema_field.name, expected_dtype, value.dtype))

            expected_shape = unischema_field.shape
            if not _is_compliant_shape(value.shape, expected_shape):
                raise ValueError('Unexpected dimensions of {} feature. '
                                 'Expected {}. Got {}'.format(unischema_field.name, expected_shape, value.shape))
        else:
            raise ValueError('Unexpected type of {} feature. '
                             'Expected ndarray of {}. Got {}'.format(unischema_field.name, expected_dtype, type(value)))

        memfile = BytesIO()
        np.savez_compressed(memfile, arr=value)
        return bytearray(memfile.getvalue())

    def decode(self, unischema_field, value):
        memfile = BytesIO(value)
        return np.load(memfile)['arr']

    def spark_dtype(self):
        return BinaryType()


class ScalarCodec(DataframeColumnCodec):
    """Encodes a scalar into a spark dataframe field."""

    def __init__(self, spark_type):
        """Constructs a codec.

        :param spark_type: an instance of a Type object from :mod:`pyspark.sql.types`
        """
        self._spark_type = spark_type

    def encode(self, unischema_field, value):
        if unischema_field.shape:
            raise ValueError('The shape field of unischema_field \'%s\' must be an empty tuple (i.e. \'()\' '
                             'to indicate a scalar. However, the actual shape is %s',
                             unischema_field.name, unischema_field.shape)
        if isinstance(self._spark_type, (ByteType, ShortType, IntegerType, LongType)):
            return int(value)
        if isinstance(self._spark_type, (FloatType, DoubleType)):
            return float(value)
        if isinstance(self._spark_type, BooleanType):
            return bool(value)
        if isinstance(self._spark_type, StringType):
            if not isinstance(value, str):
                raise ValueError(
                    'Expected a string value for field {}. Got type {}'.format(unischema_field.name, type(value)))
        return value

    def decode(self, unischema_field, value):
        # We are using pyarrow.serialize that does not support Decimal field types.
        # Tensorflow does not support Decimal types neither. We convert all decimals to
        # strings hence prevent Decimals from getting into anywhere in the reader. We may
        # choose to resurrect Decimals support in the future.
        return unischema_field.numpy_dtype(value)

    def spark_dtype(self):
        return self._spark_type


def _is_compliant_shape(a, b):
    """Compares the shapes of two arguments.

    If size of a dimensions is None, this dimension size is ignored.

    Example:

    >>> assert _is_compliant_shape((1, 2, 3), (1, 2, 3))
    >>> assert _is_compliant_shape((1, 2, 3), (1, None, 3))
    >>> assert not _is_compliant_shape((1, 2, 3), (1, 10, 3))
    >>> assert not _is_compliant_shape((1, 2), (1,))

    :return: True, if the shapes are compliant
    """
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] and b[i]:
            if a[i] != b[i]:
                return False
    return True
