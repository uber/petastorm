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
import pyarrow as pa
from pyspark import Row
from pyspark.sql.types import StringType, IntegerType, DecimalType, ShortType, LongType

from petastorm.codecs import ScalarCodec, NdarrayCodec
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row, \
    insert_explicit_nulls, match_unischema_fields

try:
    from unittest import mock
except ImportError:
    from mock import mock


def _mock_parquet_dataset(partition_names, schema):
    """Creates a pyarrow.ParquetDataset mock capable of returning:

        parquet_dataset.pieces[0].get_metadata(parquet_dataset.fs.open).schema.to_arrow_schema() == schema
        parquet_dataset.partition_names.partition_names = partition_names

    """
    piece_mock = mock.Mock()
    piece_mock.get_metadata().schema.to_arrow_schema.return_value = schema

    dataset_mock = mock.Mock()
    type(dataset_mock).pieces = mock.PropertyMock(return_value=[piece_mock])
    type(dataset_mock.partitions).partition_names = mock.PropertyMock(return_value=partition_names)

    return dataset_mock


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

    def test_create_schema_view_using_invalid_type(self):
        """ Exercises code paths unischema.create_schema_view ValueError, and unischema.__str__."""
        TestSchema = Unischema('TestSchema', [
            UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
            UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
        ])
        with self.assertRaises(ValueError) as ex:
            TestSchema.create_schema_view([42])
        self.assertTrue('must be either a string' in str(ex.exception))

    def test_create_schema_view_using_unischema_fields(self):
        TestSchema = Unischema('TestSchema', [
            UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
            UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
        ])
        view = TestSchema.create_schema_view([TestSchema.int_field])
        self.assertEqual(set(view.fields.keys()), {'int_field'})

    def test_create_schema_view_using_regex(self):
        TestSchema = Unischema('TestSchema', [
            UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
            UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
        ])
        view = TestSchema.create_schema_view(['int.*$'])
        self.assertEqual(set(view.fields.keys()), {'int_field'})

        view = TestSchema.create_schema_view([u'int.*$'])
        self.assertEqual(set(view.fields.keys()), {'int_field'})

    def test_create_schema_view_using_regex_and_unischema_fields(self):
        TestSchema = Unischema('TestSchema', [
            UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
            UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
            UnischemaField('other_string_field', np.string_, (), ScalarCodec(StringType()), False),
        ])
        view = TestSchema.create_schema_view(['int.*$', TestSchema.string_field])
        self.assertEqual(set(view.fields.keys()), {'int_field', 'string_field'})

    def test_create_schema_view_no_field_matches_regex(self):
        TestSchema = Unischema('TestSchema', [
            UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
            UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
        ])
        view = TestSchema.create_schema_view(['bogus'])
        self.assertEqual(len(view.fields), 0)

    def test_name_property(self):
        TestSchema = Unischema('TestSchema', [
            UnischemaField('nullable', np.int32, (), ScalarCodec(StringType()), True),
        ])

        self.assertEqual('TestSchema', TestSchema.name)

    def test_filter_schema_fields_from_url(self):
        TestSchema = Unischema('TestSchema', [
            UnischemaField('int32', np.int32, (), None, False),
            UnischemaField('uint8', np.uint8, (), None, False),
            UnischemaField('uint16', np.uint16, (), None, False),
        ])

        assert match_unischema_fields(TestSchema, ['.*nt.*6']) == [TestSchema.uint16]
        assert match_unischema_fields(TestSchema, ['nomatch']) == []
        assert match_unischema_fields(TestSchema, ['.*']) == list(TestSchema.fields.values())
        assert match_unischema_fields(TestSchema, ['int32', 'uint8']) == [TestSchema.int32, TestSchema.uint8]

    def test_arrow_schema_convertion(self):

        arrow_schema = pa.schema([
            pa.field('string', pa.string()),
            pa.field('int8', pa.int8()),
            pa.field('int16', pa.int16()),
            pa.field('int32', pa.int32()),
            pa.field('int64', pa.int64()),
            pa.field('float', pa.float32()),
            pa.field('double', pa.float64()),
            pa.field('bool', pa.bool_(), False),
            pa.field('fixed_size_binary', pa.binary(10)),
            pa.field('variable_size_binary', pa.binary()),
            pa.field('decimal', pa.decimal128(3, 4)),
            pa.field('timestamp_s', pa.timestamp('s')),
            pa.field('timestamp_ns', pa.timestamp('ns')),
            pa.field('date_32', pa.date32()),
            pa.field('date_64', pa.date64()),
            pa.field('timestamp_ns', pa.timestamp('ns')),
        ])

        mock_dataset = _mock_parquet_dataset([], arrow_schema)

        unischema = Unischema.from_arrow_schema(mock_dataset)
        for name in arrow_schema.names:
            assert getattr(unischema, name).name == name
            assert isinstance(getattr(unischema, name).codec, ScalarCodec)
            if name == 'bool':
                assert not getattr(unischema, name).nullable
            else:
                assert getattr(unischema, name).nullable

    def test_arrow_schema_convertion_with_partitions(self):

        arrow_schema = pa.schema([
            pa.field('int8', pa.int8()),
        ])

        mock_dataset = _mock_parquet_dataset(['part_name'], arrow_schema)

        unischema = Unischema.from_arrow_schema(mock_dataset)
        assert unischema.part_name.codec.spark_dtype().typeName() == 'string'

    def test_arrow_schema_convertion_fail(self):
        arrow_schema = pa.schema([
            pa.field('list_of_int', pa.float16()),
        ])

        mock_dataset = _mock_parquet_dataset([], arrow_schema)

        with self.assertRaises(ValueError) as ex:
            Unischema.from_arrow_schema(mock_dataset)

        assert 'Cannot auto-create unischema due to unsupported column type' in str(ex.exception)

    def test_arrow_schema_arrow_1644_list_of_struct(self):
        arrow_schema = pa.schema([
            pa.field('id', pa.string()),
            pa.field('list_of_struct', pa.list_(pa.struct([('a', pa.string()), ('b', pa.int32())])))
        ])

        mock_dataset = _mock_parquet_dataset([], arrow_schema)

        unischema = Unischema.from_arrow_schema(mock_dataset)
        assert getattr(unischema, 'id').name == 'id'
        assert not hasattr(unischema, 'list_of_struct')

    def test_arrow_schema_arrow_1644_list_of_list(self):
        arrow_schema = pa.schema([
            pa.field('id', pa.string()),
            pa.field('list_of_list',
                     pa.list_(pa.list_(pa.struct([('a', pa.string()), ('b', pa.int32())]))))
        ])

        mock_dataset = _mock_parquet_dataset([], arrow_schema)

        unischema = Unischema.from_arrow_schema(mock_dataset)
        assert getattr(unischema, 'id').name == 'id'
        assert not hasattr(unischema, 'list_of_list')

    def test_arrow_schema_convertion_ignore(self):
        arrow_schema = pa.schema([
            pa.field('list_of_int', pa.float16()),
            pa.field('struct', pa.struct([('a', pa.string()), ('b', pa.int32())])),
        ])

        mock_dataset = _mock_parquet_dataset([], arrow_schema)

        unischema = Unischema.from_arrow_schema(mock_dataset, omit_unsupported_fields=True)
        assert not hasattr(unischema, 'list_of_int')


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
