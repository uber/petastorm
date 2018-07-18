#
# Uber, Inc. (c) 2018
#
from decimal import Decimal
import numpy as np


class ReaderMock(object):
    """Reads a unischema based mock dataset."""

    def __init__(self, schema, schema_data_generator, sequence=None):
        """Initializes a reader object.

        :param schema: unischema instance
        :param schema_data_generator: A function that takes names of fields in unischema and returns the actual
                values that complies with the schema.
        """
        self.schema = schema
        self.schema_data_generator = schema_data_generator
        if sequence is not None:
            raise ValueError('Sequence argument not supported for ReaderMock')
        self.sequence = sequence

    def fetch(self):
        """
        Generates the mock dataset based on the schema.

        :return: named tuple data according to schema.
        """
        fields_as_dict = self.schema_data_generator(self.schema)
        return self.schema.make_namedtuple(**fields_as_dict)

    def __iter__(self):
        return self

    def next(self):
        return self.fetch()

    # Functions needed to treat reader as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def stop(self):
        pass

    def join(self):
        pass


def schema_data_generator_example(schema):
    """
    Generates dummy data for a given schema.

    :param schema: unischema instance
    :return: A dictionary of schema dummy values.
    """
    fields_as_dict = {}
    for field in schema.fields.values():
        if field.numpy_dtype is Decimal:
            fields_as_dict[field.name] = Decimal('0.0')
        else:
            field_shape = tuple([10 if dim is None else dim for dim in field.shape])
            if field.numpy_dtype == np.string_:
                if field_shape == ():
                    default_val = 'default'
                else:
                    default_val = ['default'] * field_shape[0]
                fields_as_dict[field.name] = np.array(default_val, dtype=field.numpy_dtype)
            else:
                fields_as_dict[field.name] = np.zeros(field_shape, dtype=field.numpy_dtype)
    return fields_as_dict
