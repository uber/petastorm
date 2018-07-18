import numpy as np
from numpy.random import rand

LIST_SIZE = 13


def generate_datapoint(schema):
    """Generates random data point following a schema

    :param schema: an instance of Unischema specifying the columns of the dataset (with name, dtype, shape and codec)

    :returns: a randomly generated datapoint with the fields and format specified by schema
    """

    # Init dict representing datapoint
    d = {}

    for key, f in schema.fields.items():
        dtype = f.numpy_dtype
        shape = tuple(d if d is not None else LIST_SIZE for d in f.shape)

        # Extract range information from data type
        min, max = 0, 1

        if issubclass(dtype, np.integer):
            min, max = np.iinfo(dtype).min, np.iinfo(dtype).max

        spread = max - min

        value = rand(*shape) * spread + min
        d[key] = np.array(value, dtype=dtype)

    return d
