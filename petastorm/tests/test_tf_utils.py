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
from __future__ import division

from collections import namedtuple
from decimal import Decimal

import numpy as np
import pytest
import tensorflow as tf

from petastorm.ngram import NGram
from petastorm.reader import Reader
from petastorm.tests.test_common import TestSchema
from petastorm.tf_utils import _sanitize_field_tf_types, _numpy_to_tf_dtypes, \
    _schema_to_tf_dtypes, tf_tensors
from petastorm.unischema import Unischema, UnischemaField
from petastorm.workers_pool.dummy_pool import DummyPool


def test_empty_dict():
    # Check two types that should be promoted/converted (uint16 and Decimal) and one that should not be
    # modified (int32)
    sample_input_dict = {
        'int32': np.asarray([-2 ** 31, 0, 100, 2 ** 31 - 1], dtype=np.int32),
        'uint16': np.asarray([0, 2, 2 ** 16 - 1], dtype=np.uint16),
        'Decimal': Decimal(1234) / Decimal(10),
    }

    TestNamedTuple = namedtuple('TestNamedTuple', sample_input_dict.keys())
    sample_input_tuple = TestNamedTuple(**sample_input_dict)
    sanitized_tuple = _sanitize_field_tf_types(sample_input_tuple)

    np.testing.assert_equal(sanitized_tuple.int32.dtype, np.int32)
    np.testing.assert_equal(sanitized_tuple.uint16.dtype, np.int32)
    assert isinstance(sanitized_tuple.Decimal, str)

    np.testing.assert_equal(sanitized_tuple.int32, sample_input_dict['int32'])
    np.testing.assert_equal(sanitized_tuple.uint16, sample_input_dict['uint16'])
    np.testing.assert_equal(str(sanitized_tuple.Decimal), str(sample_input_dict['Decimal'].normalize()))


def test_decimal_conversion():
    assert _numpy_to_tf_dtypes(Decimal) == tf.string


def test_uint16_promotion_to_int32():
    assert _numpy_to_tf_dtypes(np.uint16) == tf.int32


def test_unknown_type():
    with pytest.raises(ValueError):
        _numpy_to_tf_dtypes(np.uint64)


def test_schema_to_dtype_list():
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int32', np.int32, (), None, False),
        UnischemaField('uint8', np.uint8, (), None, False),
        UnischemaField('uint16', np.uint16, (), None, False),
        UnischemaField('Decimal', Decimal, (), None, False),
    ])

    actual_tf_dtype_list = _schema_to_tf_dtypes(TestSchema)
    # Note that the order of the fields is defined by alphabetical order of keys and always sorted by Unischema
    # to avoid ambiguity
    #  [Decimal,   int32,    uint16,   uint8] <- alphabetical order
    #  [tf.string, tf.int32, tf.int32, tf.uint8]
    np.testing.assert_equal(actual_tf_dtype_list, [tf.string, tf.int32, tf.int32, tf.uint8])


def _read_from_tf_tensors(synthetic_dataset, count, shuffling_queue_capacity, min_after_dequeue, ngram):
    """Used by several test cases. Reads a 'count' rows using reader.

    The reader is configured without row-group shuffling and guarantees deterministic order of rows up to the
    results queue TF shuffling which is controlled by 'shuffling_queue_capacity', 'min_after_dequeue' arguments.

    The function returns a tuple with: (actual data read from the dataset, a TF tensor returned by the reader)
    """

    # Nullable fields can not be read by tensorflow (what would be the dimension of a tensor for null data?)
    fields = set(TestSchema.fields.values()) - {TestSchema.matrix_nullable, TestSchema.string_array_nullable}
    schema_fields = (fields if ngram is None else ngram)

    reader = Reader(schema_fields=schema_fields, dataset_url=synthetic_dataset.url, reader_pool=DummyPool(),
                    shuffle=False)

    row_tensors = tf_tensors(reader, shuffling_queue_capacity=shuffling_queue_capacity,
                             min_after_dequeue=min_after_dequeue)

    # Read a bunch of entries from the dataset and compare the data to reference
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, start=True)

        # Collect all the data we need from 'count' number of reads
        rows_data = [sess.run(row_tensors) for _ in range(count)]

        coord.request_stop()
        coord.join(threads)

    reader.stop()
    reader.join()

    return rows_data, row_tensors


def _assert_all_tensors_have_shape(row_tensors):
    """Asserts that all elements in row_tensors list/tuple have static shape."""
    for column in row_tensors:
        assert column.get_shape().dims is not None


def _assert_expected_rows_data(synthetic_dataset, rows_data):
    """Asserts all elements of rows_data list of rows match reference data used to create the dataset"""
    for row_tuple in rows_data:

        # It is easier to work with dict as we will be indexing column names using strings
        row = row_tuple._asdict()

        # Find corresponding row in the reference data
        expected = next(d for d in synthetic_dataset.data if d['id'] == row['id'])

        # Check equivalence of all values between a checked row and a row from reference data
        for column_name, actual in row.items():
            expected_val = expected[column_name]
            if isinstance(expected_val, Decimal) or isinstance(expected_val, str):
                # Tensorflow returns all strings as bytes in python3. So we will need to decode it
                actual = actual.decode()
            elif isinstance(expected_val, np.ndarray) and expected_val.dtype.type == np.unicode_:
                actual = np.array([item.decode() for item in actual])

            if isinstance(expected_val, Decimal):
                np.testing.assert_equal(expected_val, Decimal(actual))
            else:
                np.testing.assert_equal(expected_val, actual)


@pytest.mark.forked
def test_simple_read_tensorflow(synthetic_dataset):
    """Read couple of rows. Make sure all tensors have static shape sizes assigned and the data matches reference
    data"""
    rows_data, row_tensors = \
        _read_from_tf_tensors(synthetic_dataset, count=30, shuffling_queue_capacity=0, min_after_dequeue=0, ngram=None)

    # Make sure we have static shape info for all fields
    _assert_all_tensors_have_shape(row_tensors)
    _assert_expected_rows_data(synthetic_dataset, rows_data)


@pytest.mark.forked
def test_shuffling_queue(synthetic_dataset):
    """Read data without tensorflow shuffling queue and with it. Check the the order is deterministic within
    unshuffled read and is random with shuffled read"""
    unshuffled_1, _ = _read_from_tf_tensors(synthetic_dataset, 30, shuffling_queue_capacity=0, min_after_dequeue=0,
                                            ngram=None)
    unshuffled_2, _ = _read_from_tf_tensors(synthetic_dataset, 30, shuffling_queue_capacity=0, min_after_dequeue=0,
                                            ngram=None)

    shuffled_1, shuffled_1_row_tensors = \
        _read_from_tf_tensors(synthetic_dataset, 30, shuffling_queue_capacity=10, min_after_dequeue=9, ngram=None)
    shuffled_2, _ = \
        _read_from_tf_tensors(synthetic_dataset, 30, shuffling_queue_capacity=10, min_after_dequeue=9, ngram=None)

    # Make sure we have static shapes and the data matches reference data (important since a different code path
    # is executed within tf_tensors when shuffling is specified
    _assert_all_tensors_have_shape(shuffled_1_row_tensors)
    _assert_expected_rows_data(synthetic_dataset, shuffled_1)

    assert [f.id for f in unshuffled_1] == [f.id for f in unshuffled_2]
    assert [f.id for f in unshuffled_1] != [f.id for f in shuffled_2]
    assert [f.id for f in shuffled_1] != [f.id for f in shuffled_2]


@pytest.mark.forked
def test_simple_ngram_read_tensorflow(synthetic_dataset):
    """Read a single ngram. Make sure all shapes are set and the data read matches reference data"""
    fields = {
        0: [TestSchema.id],
        1: [TestSchema.id],
        2: [TestSchema.id]
    }

    # Expecting delta between ids to be 1. Setting 1.5 as upper bound
    ngram = NGram(fields=fields, delta_threshold=1.5, timestamp_field=TestSchema.id)

    ngrams, row_tensors_seq = \
        _read_from_tf_tensors(synthetic_dataset, 30, shuffling_queue_capacity=0, min_after_dequeue=0, ngram=ngram)

    for row_tensors in row_tensors_seq.values():
        _assert_all_tensors_have_shape(row_tensors)

    for one_ngram_dict in ngrams:
        _assert_expected_rows_data(synthetic_dataset, one_ngram_dict.values())


@pytest.mark.forked
def test_shuffling_queue_with_ngrams(synthetic_dataset):
    """Read data without tensorflow shuffling queue and with it (no rowgroup shuffling). Read ngrams
    Check the the order is deterministic within unshuffled read and is random with shuffled read"""
    fields = {
        0: [TestSchema.id],
        1: [TestSchema.id],
        2: [TestSchema.id]
    }

    # Expecting delta between ids to be 1. Setting 1.5 as upper bound
    ngram = NGram(fields=fields, delta_threshold=1.5, timestamp_field=TestSchema.id)
    unshuffled_1, _ = _read_from_tf_tensors(synthetic_dataset, 30, shuffling_queue_capacity=0, min_after_dequeue=0,
                                            ngram=ngram)
    unshuffled_2, _ = _read_from_tf_tensors(synthetic_dataset, 30, shuffling_queue_capacity=0, min_after_dequeue=0,
                                            ngram=ngram)

    shuffled_1, shuffled_1_ngram = \
        _read_from_tf_tensors(synthetic_dataset, 20, shuffling_queue_capacity=30, min_after_dequeue=29, ngram=ngram)
    shuffled_2, _ = \
        _read_from_tf_tensors(synthetic_dataset, 20, shuffling_queue_capacity=30, min_after_dequeue=29, ngram=ngram)

    # shuffled_1_ngram is a dictionary of named tuple indexed by time:
    # {0: (tensor, tensor, tensor, ...),
    #  1: (tensor, tensor, tensor, ...),
    #  ...}
    for row_tensor in shuffled_1_ngram.values():
        _assert_all_tensors_have_shape(row_tensor)

    # shuffled_1 is a list of dictionaries of named tuples indexed by time:
    # [{0: (tensor, tensor, tensor, ...),
    #  1: (tensor, tensor, tensor, ...),
    #  ...}
    # {0: (tensor, tensor, tensor, ...),
    #  1: (tensor, tensor, tensor, ...),
    #  ...},...
    # ]
    for one_ngram_dict in shuffled_1:
        _assert_expected_rows_data(synthetic_dataset, one_ngram_dict.values())

    def flatten(list_of_ngrams):
        return [row for seq in list_of_ngrams for row in seq.values()]

    assert [f.id for f in flatten(unshuffled_1)] == [f.id for f in flatten(unshuffled_2)]

    assert [f.id for f in flatten(unshuffled_1)] != [f.id for f in flatten(shuffled_2)]
    assert [f.id for f in flatten(shuffled_1)] != [f.id for f in flatten(shuffled_2)]
