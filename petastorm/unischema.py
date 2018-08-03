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

"""'unischema' is a data structure definition which can be rendered as native schema/data-types objects
in several different python libraries. Currently supported arepyspark, tensorflow, numpy"""
from collections import namedtuple, OrderedDict

import copy
from pyspark import Row
from pyspark.sql.types import StructField, StructType


def _fields_as_tuple(field):
    """
    Common representation of UnischemaField for equality and hash operators.
    Defined outside class because the method won't be accessible otherwise.

    The only difference is that the type name of the `codec` field is returned
    so that the codec object ID won't differentiate two otherwise identifcial
    UniSchema fields.
    """
    return tuple([type(f) if f == field.codec else f for f in field])


class UnischemaField(namedtuple('UnischemaField', ['name', 'numpy_dtype', 'shape', 'codec', 'nullable'])):
    """
    A type used to describe a single field in the schema:

    - name: name of the field.
    - numpy_dtype: a numpy dtype reference
    - shape: shape of the multidimensional array. 'None' value is used to define a dimension with variable number of
             elements. E.g. (None, 3) defines a point cloud with three coordinates but unknown number of points.
    - codec: An instance of a codec object used to encode/decode data during serialization
             (e.g. CompressedImageCodec('png'))
    - nullable: Boolean indicating whether field can be None

    A field is considered immutable, so we override both equality and hash operators for consistency
    and efficiency.
    """

    def __eq__(self, other):
        """
        Comparing field objects via default namedtuple __repr__ representation doesn't work due to
        codec object ID changing when unpickled.

        Instead, compare all field attributes, but only codec type.

        Future: Give codec a mime identifier.
        """
        return _fields_as_tuple(self) == _fields_as_tuple(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(_fields_as_tuple(self))


# TODO: Changing fields in this class or the UnischemaField will break reading due to the schema being pickled next to
# the dataset on disk
class Unischema(object):
    """Describes a schema of a data structure which can be rendered as native schema/data-types objects
    in several different python libraries. Currently supported arepyspark, tensorflow, numpy.
    """

    def __init__(self, name, fields):
        """Creates an instance of a Unischema object.

        :param name: name of the schema
        :param fields: A list of UniversalSchemaColumn instances describing the fields. The order of the fields is
        not important - they are stored sorted by name internally
        """
        self._name = name
        self._fields = OrderedDict([(f.name, f) for f in sorted(fields, key=lambda t: t.name)])
        # Generates attributes named by the field names as an access syntax sugar.
        for f in fields:
            setattr(self, f.name, f)

        self._namedtuple = None

    def create_schema_view(self, fields):
        """
        Creates a new instance of the schema using a subset of fields.
        In the process, validates that all fields are part of the scheme.

        If one of the fields is not part of the schema an error is raised.

        Example:
            BBox2dSchema.create_schema_view(
                [BBox2dSchema.image_aligned_veh_z_rgb_8, BBox2dSchema.image_rgb_8])

            returns a schema, but with only two fields.

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
        if not self._namedtuple:
            self._namedtuple = namedtuple(self._name, sorted(self._fields.keys()))

        return self._namedtuple

    def __getstate__(self):
        # Exclude self._namedtuple from the serialized data. Loading this namedtuple from a pickle would fail otherwise.
        state = {k: v for k, v in self.__dict__.items()}
        state['_namedtuple'] = None
        return state

    def __str__(self):
        """
        Represent this as the following form:
          Unischema(name, [
            UnischemaField(field_name, field_numpy_dtype, field_shape, field_codec, field_nullable),
            ...
          ])
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
        """Returns as spark schema object derived from the unischema.

        Example:
            spark.createDataFrame(dataset_rows, BBox2dSchema.as_spark_schema())
        """
        schema_entries = [
            StructField(
                f.name,
                f.codec.spark_dtype(),
                f.nullable) for f in self._fields.values()]
        return StructType(schema_entries)

    def make_namedtuple(self, **kargs):
        """Returns schema as a namedtuple type intialized with arguments passed to this method

        Example: some_schema.make_namedtuple(field1=10, field2='abc')
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
    by the unischema

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
    """If input dictionary has missing fields that are nullable, this function will add the missing keys with None value

    If the fields that are missing are not nullable, a ValueError is raised

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
