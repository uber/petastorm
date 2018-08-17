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

import unittest
from decimal import Decimal

import numpy as np
from pyspark import Row
from pyspark.sql.types import StringType, IntegerType, DecimalType, ShortType, LongType

from petastorm.codecs import ScalarCodec, NdarrayCodec
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row, \
    insert_explicit_nulls


class UnischemaTest(unittest.TestCase):

    def test_fields(self):
        """Try using 'fields' getter"""
        TestSchema = Unischema('TestSchema', [
            UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
            UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
        ])

        self.assertEqual(len(TestSchema.fields), 2)
        self.assertEqual(TestSchema.fields['int_field'].name, 'int_field')
        self.assertEqual(TestSchema.fields['string_field'].name, 'string_field')

    def test_as_spark_schema(self):
        """Try using 'as_spark_schema' function"""
        TestSchema = Unischema('TestSchema', [
            UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
            UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
        ])

        spark_schema = TestSchema.as_spark_schema()
        self.assertEqual(spark_schema.fields[0].name, 'int_field')
        self.assertEqual(spark_schema.fields[1].name, 'string_field')

        self.assertEqual(TestSchema.fields['int_field'].name, 'int_field')
        self.assertEqual(TestSchema.fields['string_field'].name, 'string_field')

    def test_dict_to_spark_row_field_validation_scalar_types(self):
        """Test various validations done on data types when converting a dictionary to a spark row"""
        TestSchema = Unischema('TestSchema', [
            UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
        ])

        self.assertTrue(isinstance(dict_to_spark_row(TestSchema, {'string_field': 'abc'}), Row))

        # Not a nullable field
        with self.assertRaises(ValueError):
            isinstance(dict_to_spark_row(TestSchema, {'string_field': None}), Row)

        # Wrong field type
        with self.assertRaises(ValueError):
            isinstance(dict_to_spark_row(TestSchema, {'string_field': []}), Row)

    def test_dict_to_spark_row_field_validation_scalar_nullable(self):
        """Test various validations done on data types when converting a dictionary to a spark row"""
        TestSchema = Unischema('TestSchema', [
            UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), True),
            UnischemaField('nullable_implicitly_set', np.string_, (), ScalarCodec(StringType()), True),
        ])

        self.assertTrue(isinstance(dict_to_spark_row(TestSchema, {'string_field': None}), Row))

    def test_dict_to_spark_row_field_validation_ndarrays(self):
        """Test various validations done on data types when converting a dictionary to a spark row"""
        TestSchema = Unischema('TestSchema', [
            UnischemaField('tensor3d', np.float32, (10, 20, 30), NdarrayCodec(), False),
        ])

        self.assertTrue(isinstance(dict_to_spark_row(TestSchema,
                                                     {'tensor3d': np.zeros((10, 20, 30), dtype=np.float32)}), Row))

        # Null value into not nullable field
        with self.assertRaises(ValueError):
            isinstance(dict_to_spark_row(TestSchema, {'string_field': None}), Row)

        # Wrong dimensions
        with self.assertRaises(ValueError):
            isinstance(dict_to_spark_row(TestSchema, {'string_field': np.zeros((1, 2, 3), dtype=np.float32)}), Row)

    def test_make_named_tuple(self):
        TestSchema = Unischema('TestSchema', [
            UnischemaField('string_scalar', np.string_, (), ScalarCodec(StringType()), True),
            UnischemaField('int32_scalar', np.int32, (), ScalarCodec(ShortType()), False),
            UnischemaField('uint8_scalar', np.uint8, (), ScalarCodec(ShortType()), False),
            UnischemaField('int32_matrix', np.float32, (10, 20, 3), NdarrayCodec(), True),
            UnischemaField('decimal_scalar', Decimal, (10, 20, 3), ScalarCodec(DecimalType(10, 9)), False),
        ])

        TestSchema.make_namedtuple(string_scalar='abc', int32_scalar=10, uint8_scalar=20,
                                   int32_matrix=np.int32((10, 20, 3)), decimal_scalar=Decimal(123) / Decimal(10))

        TestSchema.make_namedtuple(string_scalar=None, int32_scalar=10, uint8_scalar=20,
                                   int32_matrix=None, decimal_scalar=Decimal(123) / Decimal(10))

    def test_insert_explicit_nulls(self):
        TestSchema = Unischema('TestSchema', [
            UnischemaField('nullable', np.int32, (), ScalarCodec(StringType()), True),
            UnischemaField('not_nullable', np.int32, (), ScalarCodec(ShortType()), False),
        ])

        # Insert_explicit_nulls to leave the dictionary as is.
        row_dict = {'nullable': 0, 'not_nullable': 1}
        insert_explicit_nulls(TestSchema, row_dict)
        self.assertEqual(len(row_dict), 2)
        self.assertEqual(row_dict['nullable'], 0)
        self.assertEqual(row_dict['not_nullable'], 1)

        # Insert_explicit_nulls to leave the dictionary as is.
        row_dict = {'nullable': None, 'not_nullable': 1}
        insert_explicit_nulls(TestSchema, row_dict)
        self.assertEqual(len(row_dict), 2)
        self.assertEqual(row_dict['nullable'], None)
        self.assertEqual(row_dict['not_nullable'], 1)

        # We are missing a nullable field here. insert_explicit_nulls should add a None entry.
        row_dict = {'not_nullable': 1}
        insert_explicit_nulls(TestSchema, row_dict)
        self.assertEqual(len(row_dict), 2)
        self.assertEqual(row_dict['nullable'], None)
        self.assertEqual(row_dict['not_nullable'], 1)

        # We are missing a not_nullable field here. Should raise an ValueError.
        row_dict = {'nullable': 0}
        with self.assertRaises(ValueError):
            insert_explicit_nulls(TestSchema, row_dict)

    def test_create_schema_view_fails_validate(self):
        """ Exercises code paths unischema.create_schema_view ValueError, and unischema.__str__."""
        TestSchema = Unischema('TestSchema', [
            UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
            UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
        ])
        with self.assertRaises(ValueError) as ex:
            TestSchema.create_schema_view([UnischemaField('id', np.int64, (), ScalarCodec(LongType()), False)])
        self.assertTrue('does not belong to the schema' in str(ex.exception))

    def test_name_property(self):
        TestSchema = Unischema('TestSchema', [
            UnischemaField('nullable', np.int32, (), ScalarCodec(StringType()), True),
        ])

        self.assertEqual('TestSchema', TestSchema.name)


class UnischemaFieldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._TestField1a = UnischemaField('random', np.string_, (), ScalarCodec(StringType()), False)
        cls._TestField1b = UnischemaField('random', np.string_, (), ScalarCodec(StringType()), False)
        cls._TestField1c = UnischemaField('Random', np.string_, (), ScalarCodec(StringType()), False)
        cls._TestField2a = UnischemaField('id', np.int32, (), ScalarCodec(ShortType()), False)
        cls._TestField2b = UnischemaField('id', np.int32, (), ScalarCodec(ShortType()), False)
        cls._TestField2c = UnischemaField('ID', np.int32, (), ScalarCodec(ShortType()), False)

    def test_equality(self):
        # Use assertTrue instead of assertEqual/assertNotEqual so we don't depend on which operator (__eq__ or __ne__)
        # actual implementation of assert uses
        self.assertTrue(self._TestField1a == self._TestField1b)
        self.assertTrue(self._TestField2a == self._TestField2b)
        self.assertTrue(self._TestField1a != self._TestField1c)
        self.assertTrue(self._TestField2a != self._TestField2c)

    def test_hash(self):
        self.assertEqual(hash(self._TestField1a), hash(self._TestField1b))
        self.assertEqual(hash(self._TestField2a), hash(self._TestField2b))
        self.assertNotEqual(hash(self._TestField1a), hash(self._TestField1c))
        self.assertNotEqual(hash(self._TestField2a), hash(self._TestField2c))


if __name__ == '__main__':
    # Delegate to the test framework.
    unittest.main()
