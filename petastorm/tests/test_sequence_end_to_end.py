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

from petastorm.reader import Reader
from petastorm.shuffle_options import ShuffleOptions
from petastorm.sequence import Sequence
from petastorm.tests.tempdir import temporary_directory
from petastorm.tests.test_common import create_test_dataset, TestSchema
from petastorm.tf_utils import tf_tensors
from petastorm.workers_pool.dummy_pool import DummyPool
from petastorm.workers_pool.thread_pool import ThreadPool


def _assert_equal_sequence(actual_sequence, expected_sequence, skip_fields):
    np.testing.assert_equal(actual_sequence.keys(), expected_sequence.keys())
    for timestep in actual_sequence:
        actual_dict = actual_sequence[timestep]._asdict()
        expected_dict = expected_sequence[timestep]._asdict()
        expected_filtered_keys = [key for key in expected_dict.keys() if key not in skip_fields]
        np.testing.assert_equal(list(actual_dict.keys()), expected_filtered_keys)
        for field_name in actual_dict:
            if skip_fields is not None and field_name in skip_fields:
                continue

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


class SequenceEndToEndDatasetToolkitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes dataset once per test. All tests in this class will use the same fake dataset."""
        # Write a fake dataset to this location
        cls._dataset_dir = mkdtemp('sequence_end_to_end_petastorm')
        cls._dataset_url = 'file://{}'.format(cls._dataset_dir)
        ROWS_COUNT = 1000
        cls._dataset_dicts = create_test_dataset(cls._dataset_url, range(ROWS_COUNT))

    @classmethod
    def tearDownClass(cls):
        # Remove everything created with "get_temp_dir"
        rmtree(cls._dataset_dir)

    def _test_continuous_sequence_tf(self, length):
        """Tests continuous sequence in tf of a certain length. Continuous here refers to
        that this reader will always return consecutive sequences due to shuffle being false
        and partition being 1.
        """

        with temporary_directory() as tmp_dir:
            tmp_url = 'file://{}'.format(tmp_dir)
            ids = range(99)
            fields = set(TestSchema.fields.values()) - {TestSchema.matrix_nullable}
            dataset_dicts = create_test_dataset(tmp_url, ids, num_files=1)
            sequence = Sequence(length=length, delta_threshold=10, timestamp_field='id')
            reader = Reader(tmp_url, schema_fields=fields, shuffle_options=ShuffleOptions(False),
                            reader_pool=ThreadPool(1), sequence=sequence)

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

                    expected_sequence = {}
                    for key in range(sequence.length):
                        expected_sequence[key] = TestSchema.make_namedtuple(**dataset_dicts[expected_id + key])

                    _assert_equal_sequence(actual, expected_sequence, skip_fields=['matrix_nullable'])
                    expected_id = expected_id + 1

            reader.stop()
            reader.join()

    def _test_continuous_sequence(self, length):
        """Test continuous sequence of a certain length. Continuous here refers to
        that this reader will always return consecutive sequences due to shuffle being false
        and partition being 1."""

        with temporary_directory() as tmp_dir:
            tmp_url = 'file://{}'.format(tmp_dir)
            ids = range(99)
            dataset_dicts = create_test_dataset(tmp_url, ids, num_files=1)

            sequence = Sequence(length=length, delta_threshold=10, timestamp_field='id')
            with Reader(tmp_url, shuffle_options=ShuffleOptions(False), reader_pool=ThreadPool(1),
                        sequence=sequence) as reader:
                expected_id = 0

                for i in range(10 - length):
                    actual = next(reader)
                    expected_sequence = {}
                    for key in range(sequence.length):
                        expected_sequence[key] = TestSchema.make_namedtuple(**dataset_dicts[expected_id + key])
                    np.testing.assert_equal(actual, expected_sequence)
                    expected_id = expected_id + 1

    def _test_noncontinuous_sequence_tf(self, length):
        """Test non continuous sequence in tf of a certain length. Non continuous here refers
        to that the reader will not necessarily return consecutive sequences because partition is more
        than one and false is true."""

        dataset_dicts = self._dataset_dicts
        fields = set(TestSchema.fields.values()) - {TestSchema.matrix_nullable}
        sequence = Sequence(length=length, delta_threshold=10, timestamp_field='id')
        reader = Reader(self._dataset_url, schema_fields=fields, reader_pool=ThreadPool(1),
                        sequence=sequence)

        readout_examples = tf_tensors(reader)

        # Make sure we have static shape info for all fields
        for timestep in readout_examples:
            for field in readout_examples[timestep]:
                self.assertIsNotNone(field.get_shape().dims)

        # Read a bunch of entries from the dataset and compare the data to reference
        with tf.Session() as sess:
            for _ in range(5):
                actual = sess.run(readout_examples)

                expected_sequence = {}
                for key in range(sequence.length):
                    expected_sequence[key] = TestSchema.make_namedtuple(**dataset_dicts[actual[0].id + key])
                _assert_equal_sequence(actual, expected_sequence, skip_fields=['matrix_nullable'])

        reader.stop()
        reader.join()

    def _test_noncontinuous_sequence(self, length):
        """Test noncontinuous sequence of a certain length. Non continuous here refers
        to that the reader will not necessarily return consecutive sequences because partition is more
        than one and false is true."""

        sequence = Sequence(length=length, delta_threshold=10, timestamp_field='id')
        with Reader(self._dataset_url, reader_pool=DummyPool(), sequence=sequence) as reader:

            for _ in range(10):
                actual = next(reader)
                expected_sequence = {}
                for key in range(sequence.length):
                    expected_sequence[key] = TestSchema.make_namedtuple(
                        **self._dataset_dicts[actual[0].id + key]
                    )
                np.testing.assert_equal(actual, expected_sequence)

    def test_sequence_basic_tf(self):
        """Tests basic sequence with no delta threshold with no shuffle and in the same partition."""
        self._test_continuous_sequence_tf(length=2)

    def test_sequence_basic(self):
        """Tests basic sequence with no delta threshold with no shuffle and in the same partition."""
        self._test_continuous_sequence(length=2)

    def test_sequence_basic_longer_tf(self):
        """Tests basic sequence with no delta threshold with no shuffle and in the same partition."""
        self._test_continuous_sequence_tf(length=5)

    def test_sequence_basic_longer(self):
        """Tests basic sequence with no delta threshold with no shuffle and in the same partition."""
        self._test_continuous_sequence(length=5)

    def test_sequence_basic_shuffle_multi_partition_tf(self):
        """Tests basic sequence with no delta threshold with shuffle and in many partitions."""
        self._test_noncontinuous_sequence_tf(length=2)

    def test_sequence_basic_shuffle_multi_partition(self):
        """Tests basic sequence with no delta threshold with shuffle and in many partitions."""
        self._test_noncontinuous_sequence(length=2)

    def test_sequence_basic_longer_shuffle_multi_partition_tf(self):
        """Tests basic sequence with no delta threshold with shuffle and in many partitions."""
        self._test_noncontinuous_sequence_tf(length=5)

    def test_sequence_basic_longer_shuffle_multi_partition(self):
        """Tests basic sequence with no delta threshold with shuffle and in many partitions."""
        self._test_noncontinuous_sequence(length=5)

    def test_sequence_delta_threshold_tf(self):
        """Test to verify that delta threshold work as expected in one partition in the same sequence
        and between consecutive sequences. delta threshold here refers that each sequence must not be
        more than delta threshold apart for the field specified by timestamp_field."""

        with temporary_directory() as tmp_dir:
            tmp_url = 'file://{}'.format(tmp_dir)
            ids = [0, 3, 8, 10, 11, 20, 23]
            fields = set(TestSchema.fields.values()) - {TestSchema.matrix_nullable}
            dataset_dicts = create_test_dataset(tmp_url, ids, num_files=1)
            sequence = Sequence(length=2, delta_threshold=4, timestamp_field='id')
            reader = Reader(tmp_url, schema_fields=fields, shuffle_options=ShuffleOptions(False),
                            reader_pool=DummyPool(), sequence=sequence)

            # Sequences expected: (0, 3), (8, 10), (10, 11)

            with tf.Session() as sess:
                readout = tf_tensors(reader)
                for timestep in readout:
                    for field in readout[timestep]:
                        self.assertIsNotNone(field.get_shape().dims)
                first_item = sess.run(readout)
                _assert_equal_sequence(
                    first_item,
                    {
                        0: TestSchema.make_namedtuple(**dataset_dicts[0]),
                        1: TestSchema.make_namedtuple(**dataset_dicts[1])
                    },
                    skip_fields=['matrix_nullable']
                )

                readout = tf_tensors(reader)
                for timestep in readout:
                    for field in readout[timestep]:
                        self.assertIsNotNone(field.get_shape().dims)
                second_item = sess.run(readout)
                _assert_equal_sequence(second_item, {
                    0: TestSchema.make_namedtuple(**dataset_dicts[3]),
                    1: TestSchema.make_namedtuple(**dataset_dicts[4]),
                }, skip_fields=['matrix_nullable'])

                readout = tf_tensors(reader)
                for timestep in readout:
                    for field in readout[timestep]:
                        self.assertIsNotNone(field.get_shape().dims)
                third_item = sess.run(readout)
                _assert_equal_sequence(third_item, {
                    0: TestSchema.make_namedtuple(**dataset_dicts[5]),
                    1: TestSchema.make_namedtuple(**dataset_dicts[6]),
                }, skip_fields=['matrix_nullable'])

                with self.assertRaises(OutOfRangeError):
                    sess.run(tf_tensors(reader))

            reader.stop()
            reader.join()

    def test_sequence_delta_threshold(self):
        """Test to verify that delta threshold work as expected in one partition in the same sequence
        and between consecutive sequences. delta threshold here refers that each sequence must not be
        more than delta threshold apart for the field specified by timestamp_field."""

        with temporary_directory() as tmp_dir:
            tmp_url = 'file://{}'.format(tmp_dir)
            ids = [0, 3, 8, 10, 11, 20, 23]
            dataset_dicts = create_test_dataset(tmp_url, ids, 1)
            sequence = Sequence(length=2, delta_threshold=4, timestamp_field='id')
            with Reader(tmp_url, shuffle_options=ShuffleOptions(False),
                        reader_pool=ThreadPool(1), sequence=sequence) as reader:

                # Sequences expected: (0, 3), (8, 10), (10, 11)

                first_item = next(reader)
                np.testing.assert_equal(first_item, {
                    0: TestSchema.make_namedtuple(**dataset_dicts[0]),
                    1: TestSchema.make_namedtuple(**dataset_dicts[1]),
                })

                second_item = next(reader)
                np.testing.assert_equal(second_item, {
                    0: TestSchema.make_namedtuple(**dataset_dicts[3]),
                    1: TestSchema.make_namedtuple(**dataset_dicts[4]),
                })

                third_item = next(reader)
                np.testing.assert_equal(third_item, {
                    0: TestSchema.make_namedtuple(**dataset_dicts[5]),
                    1: TestSchema.make_namedtuple(**dataset_dicts[6]),
                })

                with self.assertRaises(StopIteration):
                    next(reader)

    def test_sequence_delta_small_threshold_tf(self):
        """Test to verify that a small threshold work in sequences."""

        with temporary_directory() as tmp_dir:
            tmp_url = 'file://{}'.format(tmp_dir)
            ids = range(0, 99, 5)
            create_test_dataset(tmp_url, ids)

            sequence = Sequence(length=2, delta_threshold=1, timestamp_field='id')
            fields = set(TestSchema.fields.values()) - {TestSchema.matrix_nullable}
            reader = Reader(tmp_url, schema_fields=fields, reader_pool=DummyPool(), sequence=sequence)

            with tf.Session() as sess:
                with self.assertRaises(OutOfRangeError):
                    sess.run(tf_tensors(reader))

            reader.stop()
            reader.join()

    def test_sequence_delta_small_threshold(self):
        """Test to verify that a small threshold work in sequences."""

        with temporary_directory() as tmp_dir:
            tmp_url = 'file://{}'.format(tmp_dir)
            ids = range(0, 99, 5)
            create_test_dataset(tmp_url, ids)

            sequence = Sequence(length=2, delta_threshold=1, timestamp_field='id')
            with Reader(tmp_url, reader_pool=ThreadPool(10), sequence=sequence) as reader:
                with self.assertRaises(StopIteration):
                    next(reader)

    def test_sequence_validation(self):
        """Test to verify that sequence validation work as expected."""

        with self.assertRaises(ValueError):
            # length must be an int
            Sequence(length='abc', delta_threshold=5, timestamp_field='id')

        with self.assertRaises(ValueError):
            # delta threshold must be an int
            Sequence(length=5, delta_threshold='abc', timestamp_field='id')

        with self.assertRaises(ValueError):
            # timestamp_field must be a string
            Sequence(length=5, delta_threshold=5, timestamp_field=5)

        # Check some positive cases
        Sequence(length=5, delta_threshold=0.5, timestamp_field='id')
        Sequence(length=5, delta_threshold=Decimal('0.5'), timestamp_field='id')

    def test_sequence_length_1_tf(self):
        """Test to verify that sequence generalize to support length 1"""

        dataset_dicts = self._dataset_dicts
        sequence = Sequence(length=1, delta_threshold=0.012, timestamp_field='timestamp')
        fields = set(TestSchema.fields.values()) - {TestSchema.matrix_nullable}
        reader = Reader(self._dataset_url, schema_fields=fields, reader_pool=DummyPool(),
                        sequence=sequence)

        with tf.Session() as sess:
            for _ in range(10):
                actual = sess.run(tf_tensors(reader))
                _assert_equal_sequence(
                    actual,
                    {
                        0: TestSchema.make_namedtuple(**dataset_dicts[actual[0].id])
                    },
                    skip_fields=['matrix_nullable']
                )

        reader.stop()
        reader.join()

    def test_sequence_length_1(self):
        """Test to verify that sequence generalize to support length 1"""

        sequence = Sequence(length=1, delta_threshold=0.012, timestamp_field='id')
        with Reader(self._dataset_url, reader_pool=DummyPool(), sequence=sequence) as reader:
            for _ in range(10):
                actual = next(reader)
                expected = next(d for d in self._dataset_dicts if d['id'] == actual[0].id)
                np.testing.assert_equal(TestSchema.make_namedtuple(**expected), actual[0])

    def test_sequence_shuffle_drop_ratio(self):
        sequence = Sequence(length=5, delta_threshold=10, timestamp_field='id')
        with Reader(self._dataset_url,
                    shuffle_options=ShuffleOptions(False),
                    reader_pool=DummyPool(),
                    sequence=sequence) as reader:
            unshuffled = [row[0].id for row in reader]

        with Reader(self._dataset_url,
                    shuffle_options=ShuffleOptions(True, 6),
                    reader_pool=DummyPool(),
                    sequence=sequence) as reader:
            shuffled = [row[0].id for row in reader]

        self.assertEqual(len(unshuffled), len(shuffled))
        self.assertNotEqual(unshuffled, shuffled)


if __name__ == '__main__':
    # Delegate to the test framework.
    unittest.main()
