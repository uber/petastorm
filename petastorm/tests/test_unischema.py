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

from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark import Row
from pyspark.sql.types import StringType, IntegerType, DecimalType, ShortType, LongType

from petastorm.codecs import ScalarCodec, NdarrayCodec
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row, \
    insert_explicit_nulls, match_unischema_fields, _new_gt_255_compatible_namedtuple, _fullmatch

from unittest import mock


def _mock_parquet_dataset(partitions, arrow_schema):
    """Creates a pyarrow.ParquetDataset mock capable of returning:

        parquet_dataset.pieces[0].get_metadata(parquet_dataset.fs.open).schema.to_arrow_schema() == schema
        parquet_dataset.partitions = partitions

    :param partitions: expected to be a list of pa.parquet.PartitionSet
    :param arrow_schema: an instance of pa.arrow_schema to be assumed by the mock parquet dataset object.
    :return:
    """
    piece_mock = mock.Mock()
    piece_mock.get_metadata().schema.to_arrow_schema.return_value = arrow_schema

    dataset_mock = mock.Mock()
    type(dataset_mock).pieces = mock.PropertyMock(return_value=[piece_mock])
    type(dataset_mock).partitions = partitions

    return dataset_mock


def test_fields():
    """Try using 'fields' getter"""
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
        UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
    ])

    assert len(TestSchema.fields) == 2
    assert TestSchema.fields['int_field'].name == 'int_field'
    assert TestSchema.fields['string_field'].name == 'string_field'


def test_as_spark_schema():
    """Try using 'as_spark_schema' function"""
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
        UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
        UnischemaField('string_field_implicit', np.string_, ()),
    ])

    spark_schema = TestSchema.as_spark_schema()
    assert spark_schema.fields[0].name == 'int_field'

    assert spark_schema.fields[1].name == 'string_field'
    assert spark_schema.fields[1].dataType == StringType()

    assert spark_schema.fields[2].name == 'string_field_implicit'
    assert spark_schema.fields[2].dataType == StringType()

    assert TestSchema.fields['int_field'].name == 'int_field'
    assert TestSchema.fields['string_field'].name == 'string_field'


def test_as_spark_schema_unspecified_codec_type_for_non_scalars_raises():
    """Do not currently support choosing spark type automatically for non-scalar types."""
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int_vector_unspecified_codec', np.int8, (1,)),
    ])

    with pytest.raises(ValueError, match='has codec set to None'):
        TestSchema.as_spark_schema()


def test_as_spark_schema_unspecified_codec_type_unknown_scalar_type_raises():
    """We have a limited list of scalar types we can automatically map from numpy (+Decimal) types to spark types.
    Make sure that a ValueError is raised if an unknown type is used."""
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int_vector_unspecified_codec', object, ()),
    ])

    with pytest.raises(ValueError, match='Was not able to map type'):
        TestSchema.as_spark_schema()


def test_dict_to_spark_row_field_validation_scalar_types():
    """Test various validations done on data types when converting a dictionary to a spark row"""
    TestSchema = Unischema('TestSchema', [
        UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
    ])

    assert isinstance(dict_to_spark_row(TestSchema, {'string_field': 'abc'}), Row)

    # Not a nullable field
    with pytest.raises(ValueError):
        isinstance(dict_to_spark_row(TestSchema, {'string_field': None}), Row)

    # Wrong field type
    with pytest.raises(TypeError):
        isinstance(dict_to_spark_row(TestSchema, {'string_field': []}), Row)


def test_dict_to_spark_row_field_validation_scalar_nullable():
    """Test various validations done on data types when converting a dictionary to a spark row"""
    TestSchema = Unischema('TestSchema', [
        UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), True),
        UnischemaField('nullable_implicitly_set', np.string_, (), ScalarCodec(StringType()), True),
    ])

    assert isinstance(dict_to_spark_row(TestSchema, {'string_field': None}), Row)


def test_dict_to_spark_row_field_validation_ndarrays():
    """Test various validations done on data types when converting a dictionary to a spark row"""
    TestSchema = Unischema('TestSchema', [
        UnischemaField('tensor3d', np.float32, (10, 20, 30), NdarrayCodec(), False),
    ])

    assert isinstance(dict_to_spark_row(TestSchema, {'tensor3d': np.zeros((10, 20, 30), dtype=np.float32)}), Row)

    # Null value into not nullable field
    with pytest.raises(ValueError):
        isinstance(dict_to_spark_row(TestSchema, {'string_field': None}), Row)

    # Wrong dimensions
    with pytest.raises(ValueError):
        isinstance(dict_to_spark_row(TestSchema, {'string_field': np.zeros((1, 2, 3), dtype=np.float32)}), Row)


def test_dict_to_spark_row_order():
    TestSchema = Unischema('TestSchema', [
        UnischemaField('float_col', np.float64, ()),
        UnischemaField('int_col', np.int64, ()),
    ])
    row_dict = {
        TestSchema.int_col.name: 3,
        TestSchema.float_col.name: 2.0,
    }
    spark_row = dict_to_spark_row(TestSchema, row_dict)
    schema_field_names = list(TestSchema.fields)
    assert spark_row[0] == row_dict[schema_field_names[0]]
    assert spark_row[1] == row_dict[schema_field_names[1]]


def test_make_named_tuple():
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


def test_insert_explicit_nulls():
    TestSchema = Unischema('TestSchema', [
        UnischemaField('nullable', np.int32, (), ScalarCodec(StringType()), True),
        UnischemaField('not_nullable', np.int32, (), ScalarCodec(ShortType()), False),
    ])

    # Insert_explicit_nulls to leave the dictionary as is.
    row_dict = {'nullable': 0, 'not_nullable': 1}
    insert_explicit_nulls(TestSchema, row_dict)
    assert len(row_dict) == 2
    assert row_dict['nullable'] == 0
    assert row_dict['not_nullable'] == 1

    # Insert_explicit_nulls to leave the dictionary as is.
    row_dict = {'nullable': None, 'not_nullable': 1}
    insert_explicit_nulls(TestSchema, row_dict)
    assert len(row_dict) == 2
    assert row_dict['nullable'] is None
    assert row_dict['not_nullable'] == 1

    # We are missing a nullable field here. insert_explicit_nulls should add a None entry.
    row_dict = {'not_nullable': 1}
    insert_explicit_nulls(TestSchema, row_dict)
    assert len(row_dict) == 2
    assert row_dict['nullable'] is None
    assert row_dict['not_nullable'] == 1

    # We are missing a not_nullable field here. Should raise an ValueError.
    row_dict = {'nullable': 0}
    with pytest.raises(ValueError):
        insert_explicit_nulls(TestSchema, row_dict)


def test_create_schema_view_fails_validate():
    """ Exercises code paths unischema.create_schema_view ValueError, and unischema.__str__."""
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
        UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
    ])
    with pytest.raises(ValueError, match='does not belong to the schema'):
        TestSchema.create_schema_view([UnischemaField('id', np.int64, (), ScalarCodec(LongType()), False)])


def test_create_schema_view_using_invalid_type():
    """ Exercises code paths unischema.create_schema_view ValueError, and unischema.__str__."""
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
        UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
    ])
    with pytest.raises(ValueError, match='must be either a string'):
        TestSchema.create_schema_view([42])


def test_create_schema_view_using_unischema_fields():
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
        UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
    ])
    view = TestSchema.create_schema_view([TestSchema.int_field])
    assert set(view.fields.keys()) == {'int_field'}


def test_create_schema_view_using_regex():
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
        UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
    ])
    view = TestSchema.create_schema_view(['int.*$'])
    assert set(view.fields.keys()) == {'int_field'}

    view = TestSchema.create_schema_view([u'int.*$'])
    assert set(view.fields.keys()) == {'int_field'}


def test_create_schema_view_using_regex_and_unischema_fields():
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
        UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
        UnischemaField('other_string_field', np.string_, (), ScalarCodec(StringType()), False),
    ])
    view = TestSchema.create_schema_view(['int.*$', TestSchema.string_field])
    assert set(view.fields.keys()) == {'int_field', 'string_field'}


def test_create_schema_view_using_regex_and_unischema_fields_with_duplicates():
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
        UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
        UnischemaField('other_string_field', np.string_, (), ScalarCodec(StringType()), False),
    ])
    view = TestSchema.create_schema_view(['int.*$', TestSchema.int_field])
    assert set(view.fields.keys()) == {'int_field'}


def test_create_schema_view_no_field_matches_regex():
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int_field', np.int8, (), ScalarCodec(IntegerType()), False),
        UnischemaField('string_field', np.string_, (), ScalarCodec(StringType()), False),
    ])
    view = TestSchema.create_schema_view(['bogus'])
    assert not view.fields


def test_name_property():
    TestSchema = Unischema('TestSchema', [
        UnischemaField('nullable', np.int32, (), ScalarCodec(StringType()), True),
    ])

    assert 'TestSchema' == TestSchema._name


def test_field_name_conflict_with_unischema_attribute():
    # fields is an existing attribute of Unischema
    with pytest.warns(UserWarning, match='Can not create dynamic property'):
        Unischema('TestSchema', [UnischemaField('fields', np.int32, (), ScalarCodec(StringType()), True)])


def test_match_unischema_fields():
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int32', np.int32, (), None, False),
        UnischemaField('uint8', np.uint8, (), None, False),
        UnischemaField('uint16', np.uint16, (), None, False),
    ])

    assert match_unischema_fields(TestSchema, ['.*nt.*6']) == [TestSchema.uint16]
    assert match_unischema_fields(TestSchema, ['nomatch']) == []
    assert set(match_unischema_fields(TestSchema, ['.*'])) == set(TestSchema.fields.values())
    assert set(match_unischema_fields(TestSchema, ['int32', 'uint8'])) == {TestSchema.int32, TestSchema.uint8}


def test_match_unischema_fields_legacy_warning():
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int32', np.int32, (), None, False),
        UnischemaField('uint8', np.uint8, (), None, False),
        UnischemaField('uint16', np.uint16, (), None, False),
    ])

    # Check that no warnings are shown if the legacy and the new way of filtering produce the same results.
    with pytest.warns(None) as unexpected_warnings:
        match_unischema_fields(TestSchema, ['uint8'])
    assert not unexpected_warnings

    # uint8 and uint16 would have been matched using the old method, but not the new one
    with pytest.warns(UserWarning, match=r'schema_fields behavior has changed.*uint16, uint8'):
        assert match_unischema_fields(TestSchema, ['uint']) == []

    # Now, all fields will be matched, but in different order (legacy vs current). Make sure we don't issue a warning.
    with pytest.warns(None) as unexpected_warnings:
        match_unischema_fields(TestSchema, ['int', 'uint8', 'uint16', 'int32'])
    assert not unexpected_warnings


def test_arrow_schema_convertion():
    fields = [
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
        pa.field('date_64', pa.date64())
    ]
    arrow_schema = pa.schema(fields)

    mock_dataset = _mock_parquet_dataset([], arrow_schema)

    unischema = Unischema.from_arrow_schema(mock_dataset)
    for name in arrow_schema.names:
        assert getattr(unischema, name).name == name
        assert getattr(unischema, name).codec is None

        if name == 'bool':
            assert not getattr(unischema, name).nullable
        else:
            assert getattr(unischema, name).nullable

    # Test schema preserve fields order
    field_name_list = [f.name for f in fields]
    assert list(unischema.fields.keys()) == field_name_list


def test_arrow_schema_convertion_with_string_partitions():
    arrow_schema = pa.schema([
        pa.field('int8', pa.int8()),
    ])

    mock_dataset = _mock_parquet_dataset([pq.PartitionSet('part_name', ['a', 'b'])], arrow_schema)

    unischema = Unischema.from_arrow_schema(mock_dataset)
    assert unischema.part_name.numpy_dtype == np.str_


def test_arrow_schema_convertion_with_int_partitions():
    arrow_schema = pa.schema([
        pa.field('int8', pa.int8()),
    ])

    mock_dataset = _mock_parquet_dataset([pq.PartitionSet('part_name', ['0', '1', '2'])], arrow_schema)

    unischema = Unischema.from_arrow_schema(mock_dataset)
    assert unischema.part_name.numpy_dtype == np.int64


def test_arrow_schema_convertion_fail():
    arrow_schema = pa.schema([
        pa.field('list_of_int', pa.float16()),
    ])

    mock_dataset = _mock_parquet_dataset([], arrow_schema)

    with pytest.raises(ValueError, match='Cannot auto-create unischema due to unsupported column type'):
        Unischema.from_arrow_schema(mock_dataset, omit_unsupported_fields=False)


def test_arrow_schema_arrow_1644_list_of_struct():
    arrow_schema = pa.schema([
        pa.field('id', pa.string()),
        pa.field('list_of_struct', pa.list_(pa.struct([pa.field('a', pa.string()), pa.field('b', pa.int32())])))
    ])

    mock_dataset = _mock_parquet_dataset([], arrow_schema)

    unischema = Unischema.from_arrow_schema(mock_dataset)
    assert getattr(unischema, 'id').name == 'id'
    assert not hasattr(unischema, 'list_of_struct')


def test_arrow_schema_arrow_1644_list_of_list():
    arrow_schema = pa.schema([
        pa.field('id', pa.string()),
        pa.field('list_of_list',
                 pa.list_(pa.list_(pa.struct([pa.field('a', pa.string()), pa.field('b', pa.int32())]))))
    ])

    mock_dataset = _mock_parquet_dataset([], arrow_schema)

    unischema = Unischema.from_arrow_schema(mock_dataset)
    assert getattr(unischema, 'id').name == 'id'
    assert not hasattr(unischema, 'list_of_list')


def test_arrow_schema_convertion_ignore():
    arrow_schema = pa.schema([
        pa.field('list_of_int', pa.float16()),
        pa.field('struct', pa.struct([pa.field('a', pa.string()), pa.field('b', pa.int32())])),
    ])

    mock_dataset = _mock_parquet_dataset([], arrow_schema)

    unischema = Unischema.from_arrow_schema(mock_dataset, omit_unsupported_fields=True)
    assert not hasattr(unischema, 'list_of_int')


@pytest.fixture()
def equality_fields():
    class Fixture(object):
        string1 = UnischemaField('random', np.string_, (), ScalarCodec(StringType()), False)
        string2 = UnischemaField('random', np.string_, (), ScalarCodec(StringType()), False)
        string_implicit = UnischemaField('random', np.string_, ())
        string_nullable = UnischemaField('random', np.string_, (), nullable=True)
        other_string = UnischemaField('Random', np.string_, (), ScalarCodec(StringType()), False)
        int1 = UnischemaField('id', np.int32, (), ScalarCodec(ShortType()), False)
        int2 = UnischemaField('id', np.int32, (), ScalarCodec(ShortType()), False)
        other_int = UnischemaField('ID', np.int32, (), ScalarCodec(ShortType()), False)

    return Fixture()


def test_equality(equality_fields):
    # Use assertTrue instead of assertEqual/assertNotEqual so we don't depend on which operator (__eq__ or __ne__)
    # actual implementation of assert uses
    assert equality_fields.string1 == equality_fields.string2
    assert equality_fields.string1 == equality_fields.string_implicit
    assert equality_fields.int1 == equality_fields.int2
    assert equality_fields.string1 != equality_fields.other_string
    assert equality_fields.other_string != equality_fields.string_implicit
    assert equality_fields.int1 != equality_fields.other_int
    assert equality_fields.string_nullable != equality_fields.string_implicit


def test_hash(equality_fields):
    assert hash(equality_fields.string1) == hash(equality_fields.string2)
    assert hash(equality_fields.int1) == hash(equality_fields.int2)
    assert hash(equality_fields.string1) != hash(equality_fields.other_string)
    assert hash(equality_fields.int1) != hash(equality_fields.other_int)


def test_new_gt_255_compatible_namedtuple():
    fields_count = 1000
    field_names = ['f{}'.format(i) for i in range(fields_count)]
    values = list(range(1000))
    huge_tuple = _new_gt_255_compatible_namedtuple('HUGE_TUPLE', field_names)
    huge_tuple_instance = huge_tuple(**dict(zip(field_names, values)))
    assert len(huge_tuple_instance) == fields_count
    assert huge_tuple_instance.f764 == 764


def test_fullmatch():
    assert _fullmatch('abc', 'abc')
    assert _fullmatch('^abc', 'abc')
    assert _fullmatch('abc$', 'abc')
    assert _fullmatch('a.c', 'abc')
    assert _fullmatch('.*abcdef', 'abcdef')
    assert _fullmatch('abc.*', 'abcdef')
    assert _fullmatch('.*c.*', 'abcdef')
    assert _fullmatch('', '')
    assert not _fullmatch('abc', 'xyz')
    assert not _fullmatch('abc', 'abcx')
    assert not _fullmatch('abc', 'xabc')
