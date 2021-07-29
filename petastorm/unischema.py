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
from typing import Dict, Any, Tuple, Optional, NamedTuple

import numpy as np
import pyarrow as pa
import six
from pyarrow.lib import ListType
from pyarrow.lib import StructType as pyStructType
from six import string_types

# _UNISCHEMA_FIELD_ORDER available values are 'preserve_input_order' or 'alphabetical'
# Current default behavior is 'preserve_input_order', the legacy behavior is 'alphabetical', which is deprecated and
# will be removed in future versions.
_UNISCHEMA_FIELD_ORDER = 'preserve_input_order'


def _fields_as_tuple(field):
    """Common representation of UnischemaField for equality and hash operators.
    Defined outside class because the method won't be accessible otherwise.

    Today codec instance also responsible for defining spark dataframe type. This knowledge should move
    to a different class in order to support backends other than Apache Parquet. For now we ignore the codec
    in comparison. From the checks does not seem that it should negatively effect the rest of the code.
    """
    return (field.name, field.numpy_dtype, field.shape, field.nullable)


class UnischemaField(NamedTuple):
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

    name: str
    numpy_dtype: Any
    shape: Tuple[Optional[int], ...]
    codec: Optional[Any] = None
    nullable: Optional[bool] = False

    def __eq__(self, other):
        """Comparing field objects via default namedtuple __repr__ representation doesn't work due to
        codec object ID changing when unpickled.

        Instead, compare all field attributes, except for codec type.

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
    _store: Dict[str, Any] = dict()

    @staticmethod
    def get(parent_schema_name, field_names):
        """Creates a nametuple with field_names as values. Returns an existing instance if was already created.

        :param parent_schema_name: Schema name becomes is part of the cache key
        :param field_names: defines names of the fields in the namedtuple created/returned. Also part of the cache key.
        :return: A namedtuple with field names defined by `field_names`
        """
        # Cache key is a combination of schema name and all field names
        if _UNISCHEMA_FIELD_ORDER.lower() == 'alphabetical':
            field_names = list(sorted(field_names))
        else:
            field_names = list(field_names)
        key = ' '.join([parent_schema_name] + field_names)
        if key not in _NamedtupleCache._store:
            _NamedtupleCache._store[key] = _new_gt_255_compatible_namedtuple(
                '{}_view'.format(parent_schema_name), field_names)
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


def _numpy_to_spark_mapping():
    """Returns a mapping from numpy to pyspark.sql type. Caches the mapping dictionary inorder to avoid instantiation
    of multiple objects in each call."""

    # Refer to the attribute of the function we use to cache the map using a name in the variable instead of a 'dot'
    # notation to avoid copy/paste/typo mistakes
    cache_attr_name = 'cached_numpy_to_pyspark_types_map'
    if not hasattr(_numpy_to_spark_mapping, cache_attr_name):
        import pyspark.sql.types as T

        setattr(_numpy_to_spark_mapping, cache_attr_name,
                {
                    np.int8: T.ByteType(),
                    np.uint8: T.ShortType(),
                    np.int16: T.ShortType(),
                    np.uint16: T.IntegerType(),
                    np.int32: T.IntegerType(),
                    np.int64: T.LongType(),
                    np.float32: T.FloatType(),
                    np.float64: T.DoubleType(),
                    np.string_: T.StringType(),
                    np.str_: T.StringType(),
                    np.unicode_: T.StringType(),
                    np.bool_: T.BooleanType(),
                })

    return getattr(_numpy_to_spark_mapping, cache_attr_name)


# TODO: Changing fields in this class or the UnischemaField will break reading due to the schema being pickled next to
# the dataset on disk
def _field_spark_dtype(field):
    if field.codec is None:
        if field.shape == ():
            spark_type = _numpy_to_spark_mapping().get(field.numpy_dtype, None)
            if not spark_type:
                raise ValueError('Was not able to map type {} to a spark type.'.format(str(field.numpy_dtype)))
        else:
            raise ValueError('An instance of non-scalar UnischemaField \'{}\' has codec set to None. '
                             'Don\'t know how to guess a Spark type for it'.format(field.name))
    else:
        spark_type = field.codec.spark_dtype()

    return spark_type


class Unischema(object):
    """Describes a schema of a data structure which can be rendered as native schema/data-types objects
    in several different python libraries. Currently supported are pyspark, tensorflow, and numpy.
    """

    def __init__(self, name, fields):
        """Creates an instance of a Unischema object.

        :param name: name of the schema
        :param fields: a list of ``UnischemaField`` instances describing the fields. The element order in the list
            represent the schema field order.
        """
        self._name = name
        if _UNISCHEMA_FIELD_ORDER.lower() == 'alphabetical':
            fields = sorted(fields, key=lambda t: t.name)

        self._fields = OrderedDict([(f.name, f) for f in fields])
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

        If one of the fields does not exist in this schema, an error is raised.

        The example returns a schema, with field_1 and any other field matching ``other.*$`` pattern.

        >>> SomeSchema.create_schema_view(
        >>>     [SomeSchema.field_1,
        >>>      'other.*$'])

        :param fields: A list of UnischemaField objects and/or regular expressions
        :return: a new view of the original schema containing only the supplied fields
        """

        # Split fields parameter to regex pattern strings and UnischemaField objects
        regex_patterns = [field for field in fields if isinstance(field, string_types)]
        # We can not check type against UnischemaField because the artifact introduced by
        # pickling, since depickled UnischemaField are of type collections.UnischemaField
        # while withing depickling they are of petastorm.unischema.UnischemaField
        # Since UnischemaField is a tuple, we check against it since it is invariant to
        # pickling
        unischema_field_objects = [field for field in fields if isinstance(field, tuple)]
        if len(unischema_field_objects) + len(regex_patterns) != len(fields):
            raise ValueError('Elements of "fields" must be either a string (regular expressions) or '
                             'an instance of UnischemaField class.')

        # For fields that are specified as instances of Unischema: make sure that this schema contains fields
        # with these names.
        exact_field_names = [field.name for field in unischema_field_objects]
        unknown_field_names = set(exact_field_names) - set(self.fields.keys())
        if unknown_field_names:
            raise ValueError('field {} does not belong to the schema {}'.format(unknown_field_names, self))

        # Do not use instances of Unischema fields passed as an argument as it could contain codec/shape
        # info that is different from the one stored in this schema object
        exact_fields = [self._fields[name] for name in exact_field_names]
        view_fields = exact_fields + match_unischema_fields(self, regex_patterns)

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
        # Lazy loading pyspark to avoid creating pyspark dependency on data reading code path
        # (currently works only with make_batch_reader)
        import pyspark.sql.types as sql_types

        schema_entries = []
        for field in self._fields.values():
            spark_type = _field_spark_dtype(field)
            schema_entries.append(sql_types.StructField(field.name, spark_type, field.nullable))

        return sql_types.StructType(schema_entries)

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
    def from_arrow_schema(cls, parquet_dataset, omit_unsupported_fields=True):
        """
        Convert an apache arrow schema into a unischema object. This is useful for datasets of only scalars
        which need no special encoding/decoding. If there is an unsupported type in the arrow schema, it will
        throw an exception.
        When the warn_only parameter is turned to True, unsupported column types prints only warnings.

        We do not set codec field in the generated fields since all parquet fields are out-of-the-box supported
        by pyarrow and we do not need perform any custom decoding.

        :param arrow_schema: :class:`pyarrow.lib.Schema`
        :param omit_unsupported_fields: :class:`Boolean`
        :return: A :class:`Unischema` object.
        """
        meta = parquet_dataset.pieces[0].get_metadata()
        arrow_schema = meta.schema.to_arrow_schema()
        unischema_fields = []

        for partition in (parquet_dataset.partitions or []):
            if (pa.types.is_binary(partition.dictionary.type) and six.PY2) or \
                    (pa.types.is_string(partition.dictionary.type) and six.PY3):
                numpy_dtype = np.str_
            elif pa.types.is_int64(partition.dictionary.type):
                numpy_dtype = np.int64
            else:
                raise RuntimeError(('Expected partition type to be one of currently supported types: string or int64. '
                                    'Got {}').format(partition.dictionary.type))

            unischema_fields.append(UnischemaField(partition.name, numpy_dtype, (), None, False))

        for column_name in arrow_schema.names:
            arrow_field = arrow_schema.field(column_name)
            field_type = arrow_field.type
            field_shape = ()
            if isinstance(field_type, ListType):
                if isinstance(field_type.value_type, ListType) or isinstance(field_type.value_type, pyStructType):
                    warnings.warn('[ARROW-1644] Ignoring unsupported structure %r for field %r'
                                  % (field_type, column_name))
                    continue
                field_shape = (None,)
            try:
                np_type = _numpy_and_codec_from_arrow_type(field_type)
            except ValueError:
                if omit_unsupported_fields:
                    warnings.warn('Column %r has an unsupported field %r. Ignoring...'
                                  % (column_name, field_type))
                    continue
                else:
                    raise
            unischema_fields.append(UnischemaField(column_name, np_type, field_shape, None, arrow_field.nullable))
        return Unischema('inferred_schema', unischema_fields)

    def __getattr__(self, item) -> Any:
        return super().__getattribute__(item)


def dict_to_spark_row(unischema, row_dict):
    """Converts a single row into a spark Row object.

    Verifies that the data confirms with unischema definition types and encodes the data using the codec specified
    by the unischema.

    The parameters are keywords to allow use of functools.partial.

    :param unischema: an instance of Unischema object
    :param row_dict: a dictionary where the keys match name of fields in the unischema.
    :return: a single pyspark.Row object
    """

    # Lazy loading pyspark to avoid creating pyspark dependency on data reading code path
    # (currently works only with make_batch_reader)
    import pyspark

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
        if schema_field.codec:
            encoded_dict[field_name] = schema_field.codec.encode(schema_field, value) if value is not None else None
        else:
            if isinstance(value, (np.generic,)):
                encoded_dict[field_name] = value.tolist()
            else:
                encoded_dict[field_name] = value

    field_list = list(unischema.fields.keys())
    # generate a value list which match the schema column order.
    value_list = [encoded_dict[name] for name in field_list]
    # create a row by value list
    row = pyspark.Row(*value_list)
    # set row fields
    row.__fields__ = field_list
    return row


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


def _fullmatch(regex, string, flags=0):
    """Emulate python-3.4 re.fullmatch()."""
    if six.PY2:
        m = re.match(regex, string, flags=flags)
        if m and (m.span() == (0, len(string))):
            return m
    else:
        return re.fullmatch(regex, string, flags)


def match_unischema_fields(schema, field_regex):
    """Returns a list of :class:`~petastorm.unischema.UnischemaField` objects that match a regular expression.

    :param schema: An instance of a :class:`~petastorm.unischema.Unischema` object.
    :param field_regex: A list of regular expression patterns. A field is matched if the regular expression matches
      the entire field name.
    :return: A list of :class:`~petastorm.unischema.UnischemaField` instances matching at least one of the regular
      expression patterns given by ``field_regex``.
    """
    if field_regex:
        unischema_fields = set()
        legacy_unischema_fields = set()
        for pattern in field_regex:
            unischema_fields |= {field for field_name, field in schema.fields.items() if
                                 _fullmatch(pattern, field_name)}
            legacy_unischema_fields |= {field for field_name, field in schema.fields.items()
                                        if re.match(pattern, field_name)}
        if unischema_fields != legacy_unischema_fields:
            field_names = {f.name for f in unischema_fields}
            legacy_field_names = {f.name for f in legacy_unischema_fields}
            # Sorting list of diff_names so it's easier to unit-test the message
            diff_names = sorted(list((field_names | legacy_field_names) - (field_names & legacy_field_names)))
            warnings.warn('schema_fields behavior has changed. Now, regular expression pattern must match'
                          ' the entire field name. The change in the behavior affects '
                          'the following fields: {}'.format(', '.join(diff_names)))
        return list(unischema_fields)
    else:
        return []


def _numpy_and_codec_from_arrow_type(field_type):
    from pyarrow import types

    if types.is_int8(field_type):
        np_type = np.int8
    elif types.is_uint8(field_type):
        np_type = np.uint8
    elif types.is_int16(field_type):
        np_type = np.int16
    elif types.is_int32(field_type):
        np_type = np.int32
    elif types.is_int64(field_type):
        np_type = np.int64
    elif types.is_string(field_type):
        np_type = np.unicode_
    elif types.is_boolean(field_type):
        np_type = np.bool_
    elif types.is_float32(field_type):
        np_type = np.float32
    elif types.is_float64(field_type):
        np_type = np.float64
    elif types.is_decimal(field_type):
        np_type = Decimal
    elif types.is_binary(field_type):
        np_type = np.string_
    elif types.is_fixed_size_binary(field_type):
        np_type = np.string_
    elif types.is_date(field_type):
        np_type = np.datetime64
    elif types.is_timestamp(field_type):
        np_type = np.datetime64
    elif types.is_list(field_type):
        np_type = _numpy_and_codec_from_arrow_type(field_type.value_type)
    else:
        raise ValueError('Cannot auto-create unischema due to unsupported column type {}'.format(field_type))
    return np_type
