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
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from petastorm.reader import Reader, ShuffleOptions
from petastorm.ngram import NGram
from petastorm.tests.tempdir import temporary_directory
from petastorm.tests.test_common import create_test_dataset, TestSchema
from petastorm.tf_utils import tf_tensors
from petastorm.workers_pool.dummy_pool import DummyPool
from petastorm.workers_pool.thread_pool import ThreadPool


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
                                        '{0} field is different'.format(field_name)
                )
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
            TestSchema.fields.get(field) for field in TestSchema.fields if field in current_field_names]
        )
        current_dict = dataset_dicts[starting_index + index]
        new_dict = {k: current_dict[k] for k in current_dict if k in current_field_names}
        expected_ngram[key] = new_schema.make_namedtuple(**new_dict)
    return expected_ngram


class NgramEndToEndDatasetToolkitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes dataset once per test. All tests in this class will use the same fake dataset."""
        # Write a fake dataset to this location
        cls._dataset_dir = mkdtemp('ngram_end_to_end_petastorm')
        cls._dataset_url = 'file://{}'.format(cls._dataset_dir)
        ROWS_COUNT = 1000
        cls._dataset_dicts = create_test_dataset(cls._dataset_url, range(ROWS_COUNT))

    @classmethod
    def tearDownClass(cls):
        # Remove everything created with "get_temp_dir"
        rmtree(cls._dataset_dir)

    def _test_continuous_ngram_tf(self, ngram_fields):
        """Tests continuous ngram in tf of a certain length. Continuous here refers to
        that this reader will always return consecutive ngrams due to shuffle being false
        and partition being 1.
        """

        with temporary_directory() as tmp_dir:
            tmp_url = 'file://{}'.format(tmp_dir)
            ids = range(99)
            dataset_dicts = create_test_dataset(tmp_url, ids, num_files=1)
            ngram = NGram(fields=ngram_fields, delta_threshold=10, timestamp_field=TestSchema.id)
            reader = Reader(
                schema_fields=ngram,
                dataset_url=tmp_url,
                reader_pool=ThreadPool(1),
                shuffle_options=ShuffleOptions(False))

            readout_examples = tf_tensors(reader)

            # Make sure we have static shape info for all fields
            for timestep in readout_examples:
                for field in readout_examples[timestep]:
                    self.assertIsNotNone(field.get_shape().dims)

            # Read a bunch of entries from the dataset and compare the data to reference
            expected_id = 0
            with tf.Session() as sess:
                for _ in range(5):
                    actual = sess.run(readout_examples)
                    expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, expected_id)
                    _assert_equal_ngram(actual, expected_ngram)
                    expected_id = expected_id + 1

            reader.stop()
            reader.join()

    def _test_continuous_ngram(self, ngram_fields):
        """Test continuous ngram of a certain length. Continuous here refers to
        that this reader will always return consecutive ngrams due to shuffle being false
        and partition being 1."""

        with temporary_directory() as tmp_dir:
            tmp_url = 'file://{}'.format(tmp_dir)
            ids = range(99)
            dataset_dicts = create_test_dataset(tmp_url, ids, num_files=1)

            ngram = NGram(fields=ngram_fields, delta_threshold=10, timestamp_field=TestSchema.id)
            with Reader(
                    schema_fields=ngram,
                    dataset_url=tmp_url,
                    reader_pool=ThreadPool(1),
                    shuffle_options=ShuffleOptions(False)) as reader:
                expected_id = 0

                for _ in range(ngram.length):
                    actual = next(reader)
                    expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, expected_id)
                    np.testing.assert_equal(actual, expected_ngram)
                    expected_id = expected_id + 1

    def _test_noncontinuous_ngram_tf(self, ngram_fields):
        """Test non continuous ngram in tf of a certain length. Non continuous here refers
        to that the reader will not necessarily return consecutive ngrams because partition is more
        than one and false is true."""

        dataset_dicts = self._dataset_dicts
        ngram = NGram(fields=ngram_fields, delta_threshold=10, timestamp_field=TestSchema.id)
        reader = Reader(
            schema_fields=ngram,
            dataset_url=self._dataset_url,
            reader_pool=ThreadPool(1),
        )

        readout_examples = tf_tensors(reader)

        # Make sure we have static shape info for all fields
        for timestep in readout_examples:
            for field in readout_examples[timestep]:
                self.assertIsNotNone(field.get_shape().dims)

        # Read a bunch of entries from the dataset and compare the data to reference
        with tf.Session() as sess:
            for _ in range(5):
                actual = sess.run(readout_examples)
                expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, actual[min(actual.keys())].id)
                _assert_equal_ngram(actual, expected_ngram)

        reader.stop()
        reader.join()

    def _test_noncontinuous_ngram(self, ngram_fields):
        """Test noncontinuous ngram of a certain length. Non continuous here refers
        to that the reader will not necessarily return consecutive ngrams because partition is more
        than one and false is true."""

        dataset_dicts = self._dataset_dicts
        ngram = NGram(fields=ngram_fields, delta_threshold=10, timestamp_field=TestSchema.id)
        with Reader(
                schema_fields=ngram,
                dataset_url=self._dataset_url,
                reader_pool=DummyPool(),
                shuffle_options=ShuffleOptions(True, 5)) as reader:

            for _ in range(10):
                actual = next(reader)
                expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, actual[min(actual.keys())].id)
                np.testing.assert_equal(actual, expected_ngram)

    def test_ngram_basic_tf(self):
        """Tests basic ngram with no delta threshold with no shuffle and in the same partition."""
        fields = {
            -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
            0: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
        }
        self._test_continuous_ngram_tf(ngram_fields=fields)

    def test_ngram_basic(self):
        """Tests basic ngram with no delta threshold with no shuffle and in the same partition."""
        fields = {
            -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
            0: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
        }
        self._test_continuous_ngram(ngram_fields=fields)

    def test_ngram_basic_longer_tf(self):
        """Tests basic ngram with no delta threshold with no shuffle and in the same partition."""
        fields = {
            -2: [TestSchema.id, TestSchema.id2, TestSchema.matrix],
            -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png],
            0: [TestSchema.id, TestSchema.id2, TestSchema.decimal],
            1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
            2: [TestSchema.id, TestSchema.id2]
        }
        self._test_continuous_ngram_tf(ngram_fields=fields)

    def test_ngram_basic_longer(self):
        """Tests basic ngram with no delta threshold with no shuffle and in the same partition."""
        fields = {
            -2: [TestSchema.id, TestSchema.id2, TestSchema.matrix],
            -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png],
            0: [TestSchema.id, TestSchema.id2, TestSchema.decimal],
            -1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
            -2: [TestSchema.id, TestSchema.id2]
        }
        self._test_continuous_ngram(ngram_fields=fields)

    def test_ngram_basic_shuffle_multi_partition_tf(self):
        """Tests basic ngram with no delta threshold with shuffle and in many partitions."""
        fields = {
            -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
            0: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
        }
        self._test_noncontinuous_ngram_tf(ngram_fields=fields)

    def test_ngram_basic_shuffle_multi_partition(self):
        """Tests basic ngram with no delta threshold with shuffle and in many partitions."""
        fields = {
            0: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
            1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
        }
        self._test_noncontinuous_ngram(ngram_fields=fields)

    def test_ngram_basic_longer_shuffle_multi_partition_tf(self):
        """Tests basic ngram with no delta threshold with shuffle and in many partitions."""
        fields = {
            -2: [TestSchema.id, TestSchema.id2, TestSchema.matrix],
            -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png],
            0: [TestSchema.id, TestSchema.id2, TestSchema.decimal],
            1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
            2: [TestSchema.id, TestSchema.id2]
        }
        self._test_noncontinuous_ngram_tf(ngram_fields=fields)

    def test_ngram_basic_longer_shuffle_multi_partition(self):
        """Tests basic ngram with no delta threshold with shuffle and in many partitions."""
        fields = {
            -5: [TestSchema.id, TestSchema.id2, TestSchema.matrix],
            -4: [TestSchema.id, TestSchema.id2, TestSchema.image_png],
            -3: [TestSchema.id, TestSchema.id2, TestSchema.decimal],
            -2: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
            -1: [TestSchema.id, TestSchema.id2]
        }
        self._test_noncontinuous_ngram(ngram_fields=fields)

    def test_ngram_delta_threshold_tf(self):
        """Test to verify that delta threshold work as expected in one partition in the same ngram
        and between consecutive ngrams. delta threshold here refers that each ngram must not be
        more than delta threshold apart for the field specified by timestamp_field."""

        with temporary_directory() as tmp_dir:
            tmp_url = 'file://{}'.format(tmp_dir)
            ids = [0, 3, 8, 10, 11, 20, 23]
            dataset_dicts = create_test_dataset(tmp_url, ids, num_files=1)
            fields = {
                0: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
                1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
            }
            ngram = NGram(fields=fields, delta_threshold=4, timestamp_field=TestSchema.id)
            reader = Reader(
                schema_fields=ngram,
                dataset_url=tmp_url,
                reader_pool=DummyPool(),
                shuffle_options=ShuffleOptions(False))

            # Ngrams expected: (0, 3), (8, 10), (10, 11)

            with tf.Session() as sess:
                readout = tf_tensors(reader)
                for timestep in readout:
                    for field in readout[timestep]:
                        self.assertIsNotNone(field.get_shape().dims)
                first_item = sess.run(readout)
                expected_item = _get_named_tuple_from_ngram(ngram, dataset_dicts, 0)
                _assert_equal_ngram(first_item, expected_item)

                readout = tf_tensors(reader)
                for timestep in readout:
                    for field in readout[timestep]:
                        self.assertIsNotNone(field.get_shape().dims)
                second_item = sess.run(readout)
                expected_item = _get_named_tuple_from_ngram(ngram, dataset_dicts, 3)
                _assert_equal_ngram(second_item, expected_item)

                readout = tf_tensors(reader)
                for timestep in readout:
                    for field in readout[timestep]:
                        self.assertIsNotNone(field.get_shape().dims)
                third_item = sess.run(readout)
                expected_item = _get_named_tuple_from_ngram(ngram, dataset_dicts, 5)
                _assert_equal_ngram(third_item, expected_item)

                with self.assertRaises(OutOfRangeError):
                    sess.run(tf_tensors(reader))

            reader.stop()
            reader.join()

    def test_ngram_delta_threshold(self):
        """Test to verify that delta threshold work as expected in one partition in the same ngram
        and between consecutive ngrams. delta threshold here refers that each ngram must not be
        more than delta threshold apart for the field specified by timestamp_field."""

        with temporary_directory() as tmp_dir:
            tmp_url = 'file://{}'.format(tmp_dir)
            ids = [0, 3, 8, 10, 11, 20, 23]
            dataset_dicts = create_test_dataset(tmp_url, ids, 1)
            fields = {
                0: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
                1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
            }
            ngram = NGram(fields=fields, delta_threshold=4, timestamp_field=TestSchema.id)
            with Reader(
                    schema_fields=ngram,
                    dataset_url=tmp_url,
                    reader_pool=ThreadPool(1),
                    shuffle_options=ShuffleOptions(False)) as reader:

                # NGrams expected: (0, 3), (8, 10), (10, 11)

                first_item = next(reader)
                expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, 0)
                np.testing.assert_equal(first_item, expected_ngram)

                second_item = next(reader)
                expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, 3)
                np.testing.assert_equal(second_item, expected_ngram)

                third_item = next(reader)
                expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, 5)
                np.testing.assert_equal(third_item, expected_ngram)

                with self.assertRaises(StopIteration):
                    next(reader)

    def test_ngram_delta_small_threshold_tf(self):
        """Test to verify that a small threshold work in ngrams."""

        with temporary_directory() as tmp_dir:
            tmp_url = 'file://{}'.format(tmp_dir)
            ids = range(0, 99, 5)
            create_test_dataset(tmp_url, ids)

            fields = {
                0: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
                1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
            }
            ngram = NGram(fields=fields, delta_threshold=1, timestamp_field=TestSchema.id)
            reader = Reader(
                schema_fields=ngram,
                dataset_url=tmp_url,
                reader_pool=DummyPool(),
            )

            with tf.Session() as sess:
                with self.assertRaises(OutOfRangeError):
                    sess.run(tf_tensors(reader))

            reader.stop()
            reader.join()

    def test_ngram_delta_small_threshold(self):
        """Test to verify that a small threshold work in ngrams."""

        with temporary_directory() as tmp_dir:
            tmp_url = 'file://{}'.format(tmp_dir)
            ids = range(0, 99, 5)
            create_test_dataset(tmp_url, ids)

            fields = {
                0: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
                1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
            }
            ngram = NGram(fields=fields, delta_threshold=1, timestamp_field=TestSchema.id)
            with Reader(schema_fields=ngram, dataset_url=tmp_url, reader_pool=ThreadPool(10)) as reader:
                with self.assertRaises(StopIteration):
                    next(reader)

    def test_ngram_validation(self):
        """Test to verify that ngram validation work as expected."""

        fields = {
            0: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
            1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
        }

        with self.assertRaises(ValueError):
            # delta threshold must be an int
            NGram(fields=fields, delta_threshold='abc', timestamp_field=TestSchema.id)

        with self.assertRaises(ValueError):
            # timestamp_field must be a field
            NGram(fields=fields, delta_threshold=5, timestamp_field=5)

        with self.assertRaises(ValueError):
            # Fields must be a dict
            NGram(fields=[], delta_threshold=5, timestamp_field=TestSchema.id)

        with self.assertRaises(ValueError):
            # Each value in fields must be an array
            NGram(fields={0: 'test'}, delta_threshold=5, timestamp_field=TestSchema.id)

        # Check some positive cases
        NGram(fields=fields, delta_threshold=0.5, timestamp_field=TestSchema.id)
        NGram(fields=fields, delta_threshold=Decimal('0.5'), timestamp_field=TestSchema.id)


    def test_ngram_length_1_tf(self):
        """Test to verify that ngram generalize to support length 1"""
        dataset_dicts = self._dataset_dicts
        fields = {0: [TestSchema.id, TestSchema.id2]}
        ngram = NGram(fields=fields, delta_threshold=0.012, timestamp_field=TestSchema.id)
        reader = Reader(self._dataset_url, schema_fields=ngram, shuffle_options=ShuffleOptions(True, 5),
                        reader_pool=DummyPool())
        with tf.Session() as sess:
            for _ in range(10):
                actual = sess.run(tf_tensors(reader))
                expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, actual[min(actual.keys())].id)
                _assert_equal_ngram(actual, expected_ngram)

        reader.stop()
        reader.join()

    def test_ngram_length_1(self):
        """Test to verify that ngram generalize to support length 1"""
        dataset_dicts = self._dataset_dicts
        fields = {0: [TestSchema.id, TestSchema.id2]}
        ngram = NGram(fields=fields, delta_threshold=0.012, timestamp_field=TestSchema.id)
        with Reader(self._dataset_url, schema_fields=ngram, shuffle_options=ShuffleOptions(True, 3),
                    reader_pool=DummyPool()) as reader:
            for _ in range(10):
                actual = next(reader)
                expected_ngram = _get_named_tuple_from_ngram(ngram, dataset_dicts, actual[min(actual.keys())].id)
                _assert_equal_ngram(actual, expected_ngram)

    def test_non_consecutive_ngram(self):
        """Test to verify that non consecutive keys for fields argument in ngrams work."""
        fields = {
            -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
            1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
        }
        self._test_continuous_ngram_tf(ngram_fields=fields)

    def test_shuffled_fields(self):
        """Test to verify not sorted keys for fields argument in ngrams work."""
        fields = {
            2: [TestSchema.id, TestSchema.id2, TestSchema.image_png, TestSchema.matrix],
            -1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
        }
        self._test_continuous_ngram_tf(ngram_fields=fields)

    def test_ngram_shuffle_drop_ratio(self):
        """Test to verify the shuffle drop ratio work as expected."""
        fields = {
            -2: [TestSchema.id, TestSchema.id2, TestSchema.matrix],
            -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png],
            0: [TestSchema.id, TestSchema.id2, TestSchema.decimal],
            1: [TestSchema.id, TestSchema.id2, TestSchema.sensor_name],
            2: [TestSchema.id, TestSchema.id2]
        }
        ngram = NGram(fields=fields, delta_threshold=10, timestamp_field=TestSchema.id)
        with Reader(self._dataset_url,
                    schema_fields=ngram,
                    shuffle_options=ShuffleOptions(False),
                    reader_pool=DummyPool()) as reader:
            unshuffled = [row[0].id for row in reader]
        with Reader(self._dataset_url,
                    schema_fields=ngram,
                    shuffle_options=ShuffleOptions(True, 6),
                    reader_pool=DummyPool()) as reader:
            shuffled = [row[0].id for row in reader]
        self.assertEqual(len(unshuffled), len(shuffled))
        self.assertNotEqual(unshuffled, shuffled)

if __name__ == '__main__':
    # Delegate to the test framework.
    unittest.main()
