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

"""A ``unischema`` is a data structure definition which can be rendered as native schema/data-types objects
in several different python libraries. Currently supported are pyspark, tensorflow, and numpy.
"""
import copy
import re
import sys
import warnings
from collections import namedtuple, OrderedDict
from decimal import Decimal

import numpy as np
import six
from pyarrow.lib import ListType
from pyarrow.lib import StructType as pyStructType
from pyspark import Row
from pyspark.sql.types import StringType, ShortType, LongType, IntegerType, BooleanType, DoubleType, \
    ByteType, \
    FloatType, DecimalType, DateType, TimestampType
from pyspark.sql.types import StructField, StructType
from six import string_types

from petastorm.codecs import ScalarCodec


def _fields_as_tuple(field):
    """Common representation of UnischemaField for equality and hash operators.
    Defined outside class because the method won't be accessible otherwise.

    The only difference is that the type name of the ``codec`` field is returned
    so that the codec object ID won't differentiate two otherwise identifcial
    UniSchema fields.
    """
    return tuple([type(f) if f == field.codec else f for f in field])


class UnischemaField(namedtuple('UnischemaField', ['name', 'numpy_dtype', 'shape', 'codec', 'nullable'])):
    """A type used to describe a single field in the schema:

    - name: name of the field.
    - numpy_dtype: a numpy ``dtype`` reference
    - shape: shape of the multidimensional array. None value is used to define a dimension with variable number of
             elements. E.g. ``(None, 3)`` defines a point cloud with three coordinates but unknown number of points.
    - codec: An instance of a codec object used to encode/decode data during serialization
             (e.g. ``CompressedImageCodec('png')``)
    - nullable: Boolean indicating whether field can be None

    A field is considered immutable, so we override both equality and hash operators for consistency
    and efficiency.
    """

    def __eq__(self, other):
        """Comparing field objects via default namedtuple __repr__ representation doesn't work due to
        codec object ID changing when unpickled.

        Instead, compare all field attributes, but only codec type.

        Future: Give codec a mime identifier.
        """
        return _fields_as_tuple(self) == _fields_as_tuple(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(_fields_as_tuple(self))


class _NamedtupleCache(object):
    """_NamedtupleCache makes sure the same instance of a namedtuple is returned for a given schema and a set of
     fields. This makes comparison between types possible. For example, `tf.data.Dataset.concatenate` implementation
     compares types to make sure two datasets can be concatenated."""
    _store = dict()

    @staticmethod
    def get(parent_schema_name, field_names):
        """Creates a nametuple with field_names as values. Returns an existing instance if was already created.

        :param parent_schema_name: Schema name becomes is part of the cache key
        :param field_names: defines names of the fields in the namedtuple created/returned. Also part of the cache key.
        :return: A namedtuple with field names defined by `field_names`
        """
        # Cache key is a combination of schema name and all field names
        sorted_names = list(sorted(field_names))
        key = ' '.join([parent_schema_name] + sorted_names)
        if key not in _NamedtupleCache._store:
            _NamedtupleCache._store[key] = \
                _new_gt_255_compatible_namedtuple('{}_view'.format(parent_schema_name), sorted_names)
        return _NamedtupleCache._store[key]


def _new_gt_255_compatible_namedtuple(*args, **kwargs):
    # Between Python 3 - 3.6.8 namedtuple can not have more than 255 fields. We use
    # our custom version of namedtuple in these cases
    if six.PY3 and sys.version_info[1] < 7:
        # Have to hide the codeblock in namedtuple_gt_255_fields.py from Python 2 interpreter
        # as it would trigger "unqualified exec is not allowed in function" SyntaxError
        from petastorm.namedtuple_gt_255_fields import namedtuple_gt_255_fields
        namedtuple_cls = namedtuple_gt_255_fields
    else:  # Python 2 or Python 3.7 and later.
        namedtuple_cls = namedtuple

    return namedtuple_cls(*args, **kwargs)


# TODO: Changing fields in this class or the UnischemaField will break reading due to the schema being pickled next to
# the dataset on disk
class Unischema(object):
    """Describes a schema of a data structure which can be rendered as native schema/data-types objects
    in several different python libraries. Currently supported are pyspark, tensorflow, and numpy.
    """

    def __init__(self, name, fields):
        """Creates an instance of a Unischema object.

        :param name: name of the schema
        :param fields: a list of ``UnischemaField`` instances describing the fields. The order of the fields is
            not important - they are stored sorted by name internally.
        """
        self._name = name
        self._fields = OrderedDict([(f.name, f) for f in sorted(fields, key=lambda t: t.name)])
        # Generates attributes named by the field names as an access syntax sugar.
        for f in fields:
            if not hasattr(self, f.name):
                setattr(self, f.name, f)
            else:
                warnings.warn(('Can not create dynamic property {} because it conflicts with an existing property of '
                               'Unischema').format(f.name))

    def create_schema_view(self, fields):
        """Creates a new instance of the schema using a subset of fields.

        Fields can be either UnischemaField objects or regular expression patterns.

        If one of the fields is not part of the schema an error is raised.

        The example returns a schema, with field_1 and any other field matching ``other.*$`` pattern.

        >>> SomeSchema.create_schema_view(
        >>>     [SomeSchema.field_1,
        >>>      'other.*$'])

        :param fields: A list of UnischemaField objects and/or regular expressions
        :return: a new view of the original schema containing only the supplied fields
        """

        # Split fields parameter to regex pattern strings and UnischemaField objects
        regex_patterns = [f for f in fields if isinstance(f, string_types)]
        # We can not check type against UnischemaField because the artifact introduced by
        # pickling, since depickled UnischemaField are of type collections.UnischemaField
        # while withing depickling they are of petastorm.unischema.UnischemaField
        # Since UnischemaField is a tuple, we check against it since it is invariant to
        # pickling
        unischema_field_objects = [f for f in fields if isinstance(f, tuple)]
        if len(unischema_field_objects) + len(regex_patterns) != len(fields):
            raise ValueError('Elements of "fields" must be either a string (regular expressions) or '
                             'an instance of UnischemaField class.')

        view_fields = unischema_field_objects + match_unischema_fields(self, regex_patterns)

        for field in unischema_field_objects:
            # Comparing by field names. Prevoiusly was looking for `field not in self._fields.values()`, but it breaks
            # due to faulty pickling: T223683
            if field.name not in self._fields:
                raise ValueError('field {} does not belong to the schema {}'.format(field, self))

        return Unischema('{}_view'.format(self._name), view_fields)

    def _get_namedtuple(self):
        return _NamedtupleCache.get(self._name, self._fields.keys())

    def __str__(self):
        """Represent this as the following form:

        >>> Unischema(name, [
        >>>   UnischemaField(name, numpy_dtype, shape, codec, field_nullable),
        >>>   ...
        >>> ])
        """
        fields_str = ''
        for field in self._fields.values():
            fields_str += '  {}(\'{}\', {}, {}, {}, {}),\n'.format(type(field).__name__, field.name,
                                                                   field.numpy_dtype.__name__,
                                                                   field.shape, field.codec, field.nullable)
        return '{}({}, [\n{}])'.format(type(self).__name__, self._name, fields_str)

    @property
    def fields(self):
        return self._fields

    def as_spark_schema(self):
        """Returns an object derived from the unischema as spark schema.

        Example:

        >>> spark.createDataFrame(dataset_rows,
        >>>                       SomeSchema.as_spark_schema())
        """
        schema_entries = [
            StructField(
                f.name,
                f.codec.spark_dtype(),
                f.nullable) for f in self._fields.values()]
        return StructType(schema_entries)

    def make_namedtuple(self, **kargs):
        """Returns schema as a namedtuple type intialized with arguments passed to this method.

        Example:

        >>> some_schema.make_namedtuple(field1=10, field2='abc')
        """
        # TODO(yevgeni): verify types
        typed_dict = dict()
        for key in kargs.keys():
            if kargs[key] is not None:
                typed_dict[key] = kargs[key]
            else:
                typed_dict[key] = None
        return self._get_namedtuple()(**typed_dict)

    def make_namedtuple_tf(self, *args, **kargs):
        return self._get_namedtuple()(*args, **kargs)

    @classmethod
    def from_arrow_schema(cls, parquet_dataset, omit_unsupported_fields=False):
        """
        Convert an apache arrow schema into a unischema object. This is useful for datasets of only scalars
        which need no special encoding/decoding. If there is an unsupported type in the arrow schema, it will
        throw an exception.
        When the warn_only parameter is turned to True, unsupported column types prints only warnings.

        :param arrow_schema: :class:`pyarrow.lib.Schema`
        :param omit_unsupported_fields: :class:`Boolean`
        :return: A :class:`Unischema` object.
        """
        meta = parquet_dataset.pieces[0].get_metadata(parquet_dataset.fs.open)
        arrow_schema = meta.schema.to_arrow_schema()
        unischema_fields = []

        for partition_name in parquet_dataset.partitions.partition_names:
            unischema_fields.append(UnischemaField(partition_name, np.str_, (), ScalarCodec(StringType()), False))

        for column_name in arrow_schema.names:
            arrow_field = arrow_schema.field_by_name(column_name)
            field_type = arrow_field.type
            if isinstance(field_type, ListType):
                if isinstance(field_type.value_type, ListType) or isinstance(field_type.value_type, pyStructType):
                    warnings.warn('[ARROW-1644] Ignoring unsupported structure %r for field %r'
                                  % (field_type, column_name))
                    continue
            try:
                codec, np_type = _numpy_and_codec_from_arrow_type(field_type)
            except ValueError:
                if omit_unsupported_fields:
                    warnings.warn('Column %r has an unsupported field %r. Ignoring...'
                                  % (column_name, field_type))
                    continue
                else:
                    raise
            unischema_fields.append(UnischemaField(column_name, np_type, (), codec, arrow_field.nullable))
        return Unischema('inferred_schema', unischema_fields)


def dict_to_spark_row(unischema, row_dict):
    """Converts a single row into a spark Row object.

    Verifies that the data confirms with unischema definition types and encodes the data using the codec specified
    by the unischema.

    The parameters are keywords to allow use of functools.partial.

    :param unischema: an instance of Unischema object
    :param row_dict: a dictionary where the keys match name of fields in the unischema.
    :return: a single pyspark.Row object
    """
    assert isinstance(unischema, Unischema)
    # Add null fields. Be careful not to mutate the input dictionary - that would be an unexpected side effect
    copy_row_dict = copy.copy(row_dict)
    insert_explicit_nulls(unischema, copy_row_dict)

    if set(copy_row_dict.keys()) != set(unischema.fields.keys()):
        raise ValueError('Dictionary fields \n{}\n do not match schema fields \n{}'.format(
            '\n'.join(sorted(copy_row_dict.keys())), '\n'.join(unischema.fields.keys())))

    encoded_dict = {}
    for field_name, value in copy_row_dict.items():
        schema_field = unischema.fields[field_name]
        if value is None:
            if not schema_field.nullable:
                raise ValueError('Field {} is not "nullable", but got passes a None value')
        encoded_dict[field_name] = schema_field.codec.encode(schema_field, value) if value is not None else None

    return Row(**encoded_dict)


def insert_explicit_nulls(unischema, row_dict):
    """If input dictionary has missing fields that are nullable, this function will add the missing keys with
    None value.

    If the fields that are missing are not nullable, a ``ValueError`` is raised.

    :param unischema: An instance of a unischema
    :param row_dict: dictionary that would be checked for missing nullable fields. The dictionary is modified inplace.
    :return: None
    """
    for field_name, value in unischema.fields.items():
        if field_name not in row_dict:
            if value.nullable:
                row_dict[field_name] = None
            else:
                raise ValueError('Field {} is not found in the row_dict, but is not nullable.'.format(field_name))


def match_unischema_fields(schema, field_regex):
    """Returns a list of :class:`~petastorm.unischema.UnischemaField` objects that match a regular expression.

    :param schema: An instance of a :class:`~petastorm.unischema.Unischema` object.
    :param field_regex: A list of regular expression patterns.
    :return: A list of :class:`~petastorm.unischema.UnischemaField` instances matching at least one of the regular
      expression patterns given by ``field_regex``.
    """
    if field_regex:
        unischema_fields = []
        for pattern in field_regex:
            unischema_fields.extend(
                [field for field_name, field in schema.fields.items() if re.match(pattern, field_name)])
    else:
        unischema_fields = field_regex
    return unischema_fields


def _numpy_and_codec_from_arrow_type(field_type):
    from pyarrow import types

    if types.is_int8(field_type):
        np_type = np.int8
        codec = ScalarCodec(ByteType())
    elif types.is_int16(field_type):
        np_type = np.int16
        codec = ScalarCodec(ShortType())
    elif types.is_int32(field_type):
        np_type = np.int32
        codec = ScalarCodec(IntegerType())
    elif types.is_int64(field_type):
        np_type = np.int64
        codec = ScalarCodec(LongType())
    elif types.is_string(field_type):
        np_type = np.unicode_
        codec = ScalarCodec(StringType())
    elif types.is_boolean(field_type):
        np_type = np.bool_
        codec = ScalarCodec(BooleanType())
    elif types.is_float32(field_type):
        np_type = np.float32
        codec = ScalarCodec(FloatType())
    elif types.is_float64(field_type):
        np_type = np.float64
        codec = ScalarCodec(DoubleType())
    elif types.is_decimal(field_type):
        np_type = Decimal
        codec = ScalarCodec(DecimalType(field_type.precision, field_type.scale))
    elif types.is_binary(field_type):
        codec = ScalarCodec(StringType())
        np_type = np.string_
    elif types.is_fixed_size_binary(field_type):
        codec = ScalarCodec(StringType())
        np_type = np.string_
    elif types.is_date(field_type):
        np_type = np.datetime64
        codec = ScalarCodec(DateType())
    elif types.is_timestamp(field_type):
        np_type = np.datetime64
        codec = ScalarCodec(TimestampType())
    elif types.is_list(field_type):
        _, np_type = _numpy_and_codec_from_arrow_type(field_type.value_type)
        codec = None
    else:
        raise ValueError('Cannot auto-create unischema due to unsupported column type {}'.format(field_type))
    return codec, np_type
