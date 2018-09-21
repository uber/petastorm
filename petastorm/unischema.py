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
from collections import namedtuple, OrderedDict

from pyspark import Row
from pyspark.sql.types import StructField, StructType


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
            _NamedtupleCache._store[key] = namedtuple('{}_view'.format(parent_schema_name), sorted_names)
        return _NamedtupleCache._store[key]


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
            setattr(self, f.name, f)

    def create_schema_view(self, fields):
        """Creates a new instance of the schema using a subset of fields.
        In the process, validates that all fields are part of the scheme.

        If one of the fields is not part of the schema an error is raised.

        The example returns a schema, but with only two fields:

        >>> SomeSchema.create_schema_view(
        >>>     [SomeSchema.field_1,
        >>>      SomeSchema.field_4])

        :param fields: subset of fields from which to create a new schema
        :return: a new view of the original schema containing only the supplied fields
        """
        for field in fields:
            # Comparing by field names. Prevoiusly was looking for `field not in self._fields.values()`, but it breaks
            # due to faulty pickling: T223683
            if field.name not in self._fields:
                raise ValueError('field {} does not belong to the schema {}'.format(field, self))
        # TODO(yevgeni): what happens when we have several views? Is it ok to have multiple namedtuples named similarly?
        return Unischema('{}_view'.format(self._name), fields)

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

    @property
    def name(self):
        return self._name

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
