# pylint: disable=bad-continuation
# Disabling lint bad-continuation due to lint issues between python 2.7 and python 3.6

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
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from petastorm import make_reader
from petastorm.ngram import NGram
from petastorm.reader_impl.same_thread_executor import SameThreadExecutor
from petastorm.tests.conftest import SyntheticDataset, maybe_cached_dataset
from petastorm.tests.test_common import create_test_dataset, TestSchema
from petastorm.tf_utils import tf_tensors

# Tests in this module will run once for each entry in the READER_FACTORIES
# pylint: disable=unnecessary-lambda
READER_FACTORIES = [
    lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs),
    lambda url, **kwargs: make_reader(url, reader_pool_type='process', workers_count=1, **kwargs),
    lambda url, **kwargs: make_reader(url, reader_engine='experimental_reader_v2', reader_pool_type='dummy',
                                      reader_engine_params={'loader_pool': SameThreadExecutor()}, **kwargs),
]


@pytest.fixture(scope="session")
def dataset_num_files_1(request, tmpdir_factory):
    def _dataset_generator():
        path = tmpdir_factory.mktemp("data").strpath
        url = 'file://' + path
        data = create_test_dataset(url, range(99), num_files=1)
        return SyntheticDataset(url=url, path=path, data=data)

    return maybe_cached_dataset(request.config, 'dataset_num_files_1', _dataset_generator)


@pytest.fixture(scope="session")
def dataset_0_3_8_10_11_20_23(request, tmpdir_factory):
    def _dataset_generator():
        path = tmpdir_factory.mktemp("data").strpath
        url = 'file://' + path
        ids = [0, 3, 8, 10, 11, 20, 23]
        data = create_test_dataset(url, ids, num_files=1)
        return SyntheticDataset(url=url, path=path, data=data)

    return maybe_cached_dataset(request.config, 'dataset_0_3_8_10_11_20_23', _dataset_generator)


@pytest.fixture(scope="session")
def dataset_range_0_99_5(request, tmpdir_factory):
    def _dataset_generator():
        path = tmpdir_factory.mktemp("data").strpath
        url = 'file://' + path
        ids = range(0, 99, 5)
        data = create_test_dataset(url, ids)
        return SyntheticDataset(url=url, path=path, data=data)

    return maybe_cached_dataset(request.config, 'dataset_range_0_99_5', _dataset_generator)


def _assert_equal_ngram(actual_ngram, expected_ngram):
    np.testing.assert_equal(sorted(actual_ngram.keys()), sorted(expected_ngram.keys()))
    for timestep in actual_ngram:
        actual_dict = actual_ngram[timestep]._asdict()
        expected_dict = expected_ngram[timestep]._asdict()
        np.testing.assert_equal(sorted(list(actual_dict.keys())), sorted(list(expected_dict.keys())))
        for field_name in actual_dict:
            actual_field = actual_dict[field_name]
            expected_field = expected_dict[field_name]

            if isinstance(expected_field, Decimal) or isinstance(expected_field, str):
                # Tensorflow returns all strings as bytes in python3. So we will need to decode it
                actual_field = actual_field.decode()
            elif isinstance(expected_field, np.ndarray) and expected_field.dtype.type == np.unicode_:
                actual_field = np.array([item.decode() for item in actual_field])

            if isinstance(expected_field, Decimal):
                np.testing.assert_equal(expected_field, Decimal(actual_field),
                                        '{0} field is different'.format(field_name))
            else:
                np.testing.assert_equal(expected_field, actual_field, '{0} field is different'.format(field_name))


def _get_named_tuple_from_ngram(ngram, dataset_dicts, starting_index):
    expected_ngram = {}
    for index, key in enumerate(range(min(ngram.fields.keys()), max(ngram.fields.keys()) + 1)):
        if key in ngram.fields:
            current_field_names = [field.name for field in ngram.fields[key]]
        else:
            current_field_names = []
        new_schema = TestSchema.create_schema_view([
            TestSchema.fields.get(field) for field in TestSchema.fields if field in current_field_names])
        current_dict = dataset_dicts[starting_index + index]
        new_dict = {k: current_dict[k] for k in current_dict if k in current_field_names}
        expected_ngram[key] = new_schema.make_namedtuple(**new_dict)
    return expected_ngram


def _test_continuous_ngram_tf(ngram_fields, dataset_num_files_1, reader_factory):
    """Tests continuous ngram in tf of a certain length. Continuous here refers to
    that this reader will always return consecutive ngrams due to shuffle being false
    and partition being 1.
    """

    ngram = NGram(fields=ngram_fields, delta_threshold=10, timestamp_field=TestSchema.id)
    with reader_factory(dataset_num_files_1.url,
                        schema_fields=ngram,
                        shuffle_row_groups=False) as reader:

        readout_examples = tf_tensors(reader)

        # Make sure we have static shape info for all fields
        for timestep in readout_examples:
            for field in readout_examples[timestep]:
                assert field.get_shape().dims is not None

        # Read a bunch of entries from the dataset and compare the data to reference
        expected_id = 0
        with tf.Session() as sess:
            for _ in range(5):
                actual = sess.run(readout_examples)
                expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_num_files_1.data, expected_id)
                _assert_equal_ngram(actual, expected_ngram)
                expected_id = expected_id + 1


def _test_continuous_ngram(ngram_fields, dataset_num_files_1, reader_factory):
    """Test continuous ngram of a certain length. Continuous here refers to
    that this reader will always return consecutive ngrams due to shuffle being false
    and partition being 1."""

    ngram = NGram(fields=ngram_fields, delta_threshold=10, timestamp_field=TestSchema.id)
    with reader_factory(dataset_num_files_1.url, schema_fields=ngram, shuffle_row_groups=False) as reader:
        expected_id = 0

        for _ in range(ngram.length):
            actual = next(reader)
            expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_num_files_1.data, expected_id)
            np.testing.assert_equal(actual, expected_ngram)
            expected_id = expected_id + 1


def _test_noncontinuous_ngram_tf(ngram_fields, synthetic_dataset, reader_factory):
    """Test non continuous ngram in tf of a certain length. Non continuous here refers
    to that the reader will not necessarily return consecutive ngrams because partition is more
    than one and false is true."""

    dataset_dicts = synthetic_dataset.data
    ngram = NGram(fields=ngram_fields, delta_threshold=10, timestamp_field=TestSchema.id)
    reader = reader_factory(synthetic_dataset.url, schema_fields=ngram)

    readout_examples = tf_tensors(reader)

    # Make sure we have static shape info for all fields
    for timestep in readout_examples:
        for field in readout_examples[timestep]:
            assert field.get_shape().dims is not None

    # Read a bunch of entries from the dataset and compare the data to reference
    with tf.Session() as sess:
        for _ in range(5):
            actual = sess.run(readout_examples)
            expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, actual[min(actual.keys())].id)
            _assert_equal_ngram(actual, expected_ngram)

    reader.stop()
    reader.join()


def _test_noncontinuous_ngram(ngram_fields, synthetic_dataset, reader_factory):
    """Test noncontinuous ngram of a certain length. Non continuous here refers
    to that the reader will not necessarily return consecutive ngrams because partition is more
    than one and false is true."""

    dataset_dicts = synthetic_dataset.data
    ngram = NGram(fields=ngram_fields, delta_threshold=10, timestamp_field=TestSchema.id)
    with reader_factory(synthetic_dataset.url,
                        schema_fields=ngram,
                        shuffle_row_groups=True,
                        shuffle_row_drop_partitions=5) as reader:
        for _ in range(10):
            actual = next(reader)
            expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, actual[min(actual.keys())].id)
            np.testing.assert_equal(actual, expected_ngram)


@pytest.mark.forked
@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_basic_tf(dataset_num_files_1, reader_factory):
    """Tests basic ngram with no delta threshold with no shuffle and in the same partition."""
    fields = {
        -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
        0: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
    }
    _test_continuous_ngram_tf(fields, dataset_num_files_1, reader_factory)


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_basic(dataset_num_files_1, reader_factory):
    """Tests basic ngram with no delta threshold with no shuffle and in the same partition."""
    fields = {
        -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
        0: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
    }
    _test_continuous_ngram(fields, dataset_num_files_1, reader_factory)


@pytest.mark.forked
@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_basic_longer_tf(dataset_num_files_1, reader_factory):
    """Tests basic ngram with no delta threshold with no shuffle and in the same partition."""
    fields = {
        -2: [TestSchema.id, TestSchema.id2, TestSchema.matrix],
        -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png],
        0: [TestSchema.id, TestSchema.id2, TestSchema.decimal],
        1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
        2: [TestSchema.id, TestSchema.id2]
    }
    _test_continuous_ngram_tf(fields, dataset_num_files_1, reader_factory)


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_basic_longer(dataset_num_files_1, reader_factory):
    """Tests basic ngram with no delta threshold with no shuffle and in the same partition."""
    fields = {
        -2: [TestSchema.id, TestSchema.id2, TestSchema.matrix],
        -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png],
        0: [TestSchema.id, TestSchema.id2, TestSchema.decimal],
        1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
        2: [TestSchema.id, TestSchema.id2]
    }
    _test_continuous_ngram(fields, dataset_num_files_1, reader_factory)


@pytest.mark.forked
@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_basic_shuffle_multi_partition_tf(synthetic_dataset, reader_factory):
    """Tests basic ngram with no delta threshold with shuffle and in many partitions."""
    fields = {
        -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
        0: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
    }
    _test_noncontinuous_ngram_tf(fields, synthetic_dataset, reader_factory)


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_basic_shuffle_multi_partition(synthetic_dataset, reader_factory):
    """Tests basic ngram with no delta threshold with shuffle and in many partitions."""
    fields = {
        0: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
        1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
    }
    _test_noncontinuous_ngram(fields, synthetic_dataset, reader_factory)


@pytest.mark.forked
@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_basic_longer_shuffle_multi_partition_tf(synthetic_dataset, reader_factory):
    """Tests basic ngram with no delta threshold with shuffle and in many partitions."""
    fields = {
        -2: [TestSchema.id, TestSchema.id2, TestSchema.matrix],
        -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png],
        0: [TestSchema.id, TestSchema.id2, TestSchema.decimal],
        1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
        2: [TestSchema.id, TestSchema.id2]
    }
    _test_noncontinuous_ngram_tf(fields, synthetic_dataset, reader_factory)


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_basic_longer_shuffle_multi_partition(synthetic_dataset, reader_factory):
    """Tests basic ngram with no delta threshold with shuffle and in many partitions."""
    fields = {
        -5: [TestSchema.id, TestSchema.id2, TestSchema.matrix],
        -4: [TestSchema.id, TestSchema.id2, TestSchema.image_png],
        -3: [TestSchema.id, TestSchema.id2, TestSchema.decimal],
        -2: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
        -1: [TestSchema.id, TestSchema.id2]
    }
    _test_noncontinuous_ngram(fields, synthetic_dataset, reader_factory)


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_basic_longer_no_overlap(synthetic_dataset, reader_factory):
    """Tests basic ngram with no delta threshold with no overlaps of timestamps."""
    fields = {
        -5: [TestSchema.id, TestSchema.id2, TestSchema.matrix],
        -4: [TestSchema.id, TestSchema.id2, TestSchema.image_png],
        -3: [TestSchema.id, TestSchema.id2, TestSchema.decimal],
        -2: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
        -1: [TestSchema.id, TestSchema.id2]
    }

    dataset_dicts = synthetic_dataset.data
    ngram = NGram(fields=fields, delta_threshold=10, timestamp_field=TestSchema.id, timestamp_overlap=False)
    with reader_factory(synthetic_dataset.url, schema_fields=ngram, shuffle_row_groups=False) as reader:
        timestamps_seen = set()
        for actual in reader:
            expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, actual[min(actual.keys())].id)
            np.testing.assert_equal(actual, expected_ngram)
            for step in actual.values():
                timestamp = step.id
                assert timestamp not in timestamps_seen
                timestamps_seen.add(timestamp)


@pytest.mark.forked
@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_delta_threshold_tf(dataset_0_3_8_10_11_20_23, reader_factory):
    """Test to verify that delta threshold work as expected in one partition in the same ngram
    and between consecutive ngrams. delta threshold here refers that each ngram must not be
    more than delta threshold apart for the field specified by timestamp_field."""

    fields = {
        0: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
        1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
    }
    ngram = NGram(fields=fields, delta_threshold=4, timestamp_field=TestSchema.id)
    with reader_factory(
            dataset_0_3_8_10_11_20_23.url,
            schema_fields=ngram,
            shuffle_row_groups=False) as reader:

        # Ngrams expected: (0, 3), (8, 10), (10, 11)

        with tf.Session() as sess:
            readout = tf_tensors(reader)
            for timestep in readout:
                for field in readout[timestep]:
                    assert field.get_shape().dims is not None
            first_item = sess.run(readout)
            expected_item = _get_named_tuple_from_ngram(ngram, dataset_0_3_8_10_11_20_23.data, 0)
            _assert_equal_ngram(first_item, expected_item)

            readout = tf_tensors(reader)
            for timestep in readout:
                for field in readout[timestep]:
                    assert field.get_shape().dims is not None
            second_item = sess.run(readout)
            expected_item = _get_named_tuple_from_ngram(ngram, dataset_0_3_8_10_11_20_23.data, 3)
            _assert_equal_ngram(second_item, expected_item)

            readout = tf_tensors(reader)
            for timestep in readout:
                for field in readout[timestep]:
                    assert field.get_shape().dims is not None
            third_item = sess.run(readout)
            expected_item = _get_named_tuple_from_ngram(ngram, dataset_0_3_8_10_11_20_23.data, 5)
            _assert_equal_ngram(third_item, expected_item)

            with pytest.raises(OutOfRangeError):
                sess.run(tf_tensors(reader))


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_delta_threshold(dataset_0_3_8_10_11_20_23, reader_factory):
    """Test to verify that delta threshold work as expected in one partition in the same ngram
    and between consecutive ngrams. delta threshold here refers that each ngram must not be
    more than delta threshold apart for the field specified by timestamp_field."""

    fields = {
        0: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
        1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
    }
    ngram = NGram(fields=fields, delta_threshold=4, timestamp_field=TestSchema.id)
    with reader_factory(dataset_0_3_8_10_11_20_23.url, schema_fields=ngram,
                        shuffle_row_groups=False) as reader:
        # NGrams expected: (0, 3), (8, 10), (10, 11)

        first_item = next(reader)
        expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_0_3_8_10_11_20_23.data, 0)
        np.testing.assert_equal(first_item, expected_ngram)

        second_item = next(reader)
        expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_0_3_8_10_11_20_23.data, 3)
        np.testing.assert_equal(second_item, expected_ngram)

        third_item = next(reader)
        expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_0_3_8_10_11_20_23.data, 5)
        np.testing.assert_equal(third_item, expected_ngram)

        with pytest.raises(StopIteration):
            next(reader)


@pytest.mark.forked
@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_delta_small_threshold_tf(reader_factory, dataset_range_0_99_5):
    """Test to verify that a small threshold work in ngrams."""

    fields = {
        0: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
        1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
    }
    ngram = NGram(fields=fields, delta_threshold=1, timestamp_field=TestSchema.id)
    with reader_factory(dataset_range_0_99_5.url, schema_fields=ngram) as reader:
        with tf.Session() as sess:
            with pytest.raises(OutOfRangeError):
                sess.run(tf_tensors(reader))


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_delta_small_threshold(reader_factory, dataset_range_0_99_5):
    """Test to verify that a small threshold work in ngrams."""

    fields = {
        0: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
        1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
    }
    ngram = NGram(fields=fields, delta_threshold=1, timestamp_field=TestSchema.id)
    with reader_factory(dataset_range_0_99_5.url, schema_fields=ngram) as reader:
        with pytest.raises(StopIteration):
            next(reader)


def test_ngram_validation():
    """Test to verify that ngram validation work as expected."""

    fields = {
        0: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
        1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
    }

    with pytest.raises(ValueError):
        # delta threshold must be an int
        NGram(fields=fields, delta_threshold='abc', timestamp_field=TestSchema.id)

    with pytest.raises(ValueError):
        # timestamp_field must be a field
        NGram(fields=fields, delta_threshold=5, timestamp_field=5)

    with pytest.raises(ValueError):
        # Fields must be a dict
        NGram(fields=[], delta_threshold=5, timestamp_field=TestSchema.id)

    with pytest.raises(ValueError):
        # Each value in fields must be an array
        NGram(fields={0: 'test'}, delta_threshold=5, timestamp_field=TestSchema.id)

    with pytest.raises(ValueError):
        # timestamp_overlap must be bool
        NGram(fields=fields, delta_threshold=0.5, timestamp_field=TestSchema.id, timestamp_overlap=2)

    # Check some positive cases
    NGram(fields=fields, delta_threshold=0.5, timestamp_field=TestSchema.id)
    NGram(fields=fields, delta_threshold=Decimal('0.5'), timestamp_field=TestSchema.id)


@pytest.mark.forked
@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_length_1_tf(synthetic_dataset, reader_factory):
    """Test to verify that ngram generalize to support length 1"""
    dataset_dicts = synthetic_dataset.data
    fields = {0: [TestSchema.id, TestSchema.id2]}
    ngram = NGram(fields=fields, delta_threshold=0.012, timestamp_field=TestSchema.id)
    reader = reader_factory(synthetic_dataset.url, schema_fields=ngram,
                            shuffle_row_groups=True, shuffle_row_drop_partitions=5)
    with tf.Session() as sess:
        for _ in range(10):
            actual = sess.run(tf_tensors(reader))
            expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, actual[min(actual.keys())].id)
            _assert_equal_ngram(actual, expected_ngram)

    reader.stop()
    reader.join()


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_length_1(synthetic_dataset, reader_factory):
    """Test to verify that ngram generalize to support length 1"""
    dataset_dicts = synthetic_dataset.data
    fields = {0: [TestSchema.id, TestSchema.id2]}
    ngram = NGram(fields=fields, delta_threshold=0.012, timestamp_field=TestSchema.id)
    with reader_factory(synthetic_dataset.url, schema_fields=ngram,
                        shuffle_row_groups=True, shuffle_row_drop_partitions=3) as reader:
        for _ in range(10):
            actual = next(reader)
            expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, actual[min(actual.keys())].id)
            _assert_equal_ngram(actual, expected_ngram)


@pytest.mark.forked
@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_non_consecutive_ngram(dataset_num_files_1, reader_factory):
    """Test to verify that non consecutive keys for fields argument in ngrams work."""
    fields = {
        -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
        1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
    }
    _test_continuous_ngram_tf(fields, dataset_num_files_1, reader_factory)


@pytest.mark.forked
@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_shuffled_fields(dataset_num_files_1, reader_factory):
    """Test to verify not sorted keys for fields argument in ngrams work."""
    fields = {
        2: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
        -1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
    }
    _test_continuous_ngram_tf(fields, dataset_num_files_1, reader_factory)


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_shuffle_drop_ratio(synthetic_dataset, reader_factory):
    """Test to verify the shuffle drop ratio work as expected."""
    fields = {
        -2: [TestSchema.id, TestSchema.id2, TestSchema.matrix],
        -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png],
        0: [TestSchema.id, TestSchema.id2, TestSchema.decimal],
        1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
        2: [TestSchema.id, TestSchema.id2]
    }
    ngram = NGram(fields=fields, delta_threshold=10, timestamp_field=TestSchema.id)
    with reader_factory(synthetic_dataset.url,
                        schema_fields=ngram,
                        shuffle_row_groups=False) as reader:
        unshuffled = [row[0].id for row in reader]
    with reader_factory(synthetic_dataset.url,
                        schema_fields=ngram,
                        shuffle_row_groups=True,
                        shuffle_row_drop_partitions=6) as reader:
        shuffled = [row[0].id for row in reader]
    assert len(unshuffled) == len(shuffled)
    assert unshuffled != shuffled


def _test_continuous_ngram_returns(ngram_fields, ts_field, dataset_num_files_1, reader_factory):
    """Test continuous ngram of a certain length. Continuous here refers to
    that this reader will always return consecutive ngrams due to shuffle being false
    and partition being 1. Returns the ngram object"""

    ngram = NGram(fields=ngram_fields, delta_threshold=10, timestamp_field=ts_field)
    with reader_factory(dataset_num_files_1.url, schema_fields=ngram, shuffle_row_groups=False) as reader:
        expected_id = 0

        for _ in range(ngram.length):
            actual = next(reader)
            expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_num_files_1.data, expected_id)
            np.testing.assert_equal(actual, expected_ngram)
            expected_id = expected_id + 1

    return ngram


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_with_regex_fields(dataset_num_files_1, reader_factory):
    """Tests to verify fields and timestamp field can be regular expressions and work with a reader
    """
    fields = {
        -1: ["^id.*$", "sensor_name", TestSchema.partition_key],
        0: ["^id.*$", "sensor_name", TestSchema.partition_key],
        1: ["^id.*$", "sensor_name", TestSchema.partition_key]
    }

    ts_field = '^id$'

    expected_fields = [TestSchema.id, TestSchema.id2, TestSchema.id_float, TestSchema.id_odd,
                       TestSchema.sensor_name, TestSchema.partition_key]

    ngram = _test_continuous_ngram_returns(fields, ts_field, dataset_num_files_1, reader_factory)

    # fields should get resolved after call to a reader
    ngram_fields = ngram.fields

    # Can't do direct set equality between expected fields and ngram.fields b/c of issue
    # with `Collections.UnischemaField` (see unischema.py for more information). __hash__
    # and __eq__ is implemented correctly for UnischemaField. However, a collections.UnischemaField
    # object will not use the __hash__ definied in `petastorm.unischema.py`
    for k in ngram_fields.keys():
        assert len(expected_fields) == len(ngram_fields[k])

        for curr_field in expected_fields:
            assert curr_field in ngram_fields[k]

    assert TestSchema.id == ngram._timestamp_field


@pytest.mark.parametrize('reader_factory', READER_FACTORIES)
def test_ngram_regex_field_resolve(dataset_num_files_1, reader_factory):
    """Tests ngram.resolve_regex_field_names function
    """
    fields = {
        -1: ["^id.*", "sensor_name", TestSchema.partition_key],
        0: ["^id.*", "sensor_name", TestSchema.partition_key],
        1: ["^id.*", "sensor_name", TestSchema.partition_key]
    }

    ts_field = '^id$'

    ngram = NGram(fields=fields, delta_threshold=10, timestamp_field=ts_field)

    expected_fields = {TestSchema.id, TestSchema.id2, TestSchema.id_float, TestSchema.id_odd,
                       TestSchema.sensor_name, TestSchema.partition_key}

    ngram.resolve_regex_field_names(TestSchema)

    ngram_fields = ngram.fields

    # Can't do direct set equality between expected fields and ngram.fields b/c of issue
    # with `Collections.UnischemaField` (see unischema.py for more information). __hash__
    # and __eq__ is implemented correctly for UnischemaField. However, a collections.UnischemaField
    # object will not use the __hash__ definied in `petastorm.unischema.py`
    for k in ngram_fields.keys():
        assert len(expected_fields) == len(ngram_fields[k])

        for curr_field in expected_fields:
            assert curr_field in ngram_fields[k]

    assert TestSchema.id == ngram._timestamp_field
