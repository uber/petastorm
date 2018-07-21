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
import tensorflow as tf
import unittest

from dataset_toolkit.test_util.reader_mock import ReaderMock, schema_data_generator_example
from dataset_toolkit.tf_utils import tf_tensors, _numpy_to_tf_dtypes
from dataset_toolkit.tests.test_end_to_end import TestSchema


class ReaderMockTest(unittest.TestCase):

    def setUp(self):
        self.reader = ReaderMock(TestSchema, schema_data_generator_example)

    def test_simple_read(self):
        """Just a bunch of read and compares of all values to the expected values for their types and
        shapes."""
        # Read a bunch of entries from the dataset and compare the data to reference
        for _ in range(10):
            actual = dict(next(self.reader)._asdict())
            for schema_field in TestSchema.fields.values():
                if schema_field.numpy_dtype == Decimal:
                    self.assertTrue(isinstance(actual[schema_field.name], Decimal))
                else:
                    self.assertTrue(actual[schema_field.name].dtype.type is schema_field.numpy_dtype)
                    self.assertEqual(len(actual[schema_field.name].shape), len(schema_field.shape))

        self.reader.stop()
        self.reader.join()

    def test_simple_read_tf(self):
        """Just a bunch of read and compares of all values to the expected values for their types
        and shapes"""
        reader_tensors = tf_tensors(self.reader)._asdict()

        for schema_field in TestSchema.fields.values():
            self.assertEqual(reader_tensors[schema_field.name].dtype,
                             _numpy_to_tf_dtypes(schema_field.numpy_dtype))
            self.assertEqual(len(reader_tensors[schema_field.name].shape), len(schema_field.shape))

        # Read a bunch of entries from the dataset and compare the data to reference
        with tf.Session() as sess:
            for _ in range(10):
                sess.run(reader_tensors)

        self.reader.stop()
        self.reader.join()


if __name__ == '__main__':
    unittest.main()
