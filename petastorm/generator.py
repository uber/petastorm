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

import numpy as np
from numpy.random import rand

LIST_SIZE = 13


def generate_datapoint(schema):
    """Generates random data point following a schema.

    :param schema: an instance of Unischema specifying the columns of the dataset (with name, dtype, shape and codec)

    :returns: a randomly generated datapoint with the fields and format specified by schema
    """

    # Init dict representing datapoint
    d = {}

    for key, f in schema.fields.items():
        dtype = f.numpy_dtype
        shape = tuple(d if d is not None else LIST_SIZE for d in f.shape)

        # Extract range information from data type
        min_val, max_val = 0, 1

        if issubclass(dtype, np.integer):
            min_val, max_val = np.iinfo(dtype).min, np.iinfo(dtype).max

        spread = max_val - min_val

        value = rand(*shape) * spread + min_val
        d[key] = np.array(value, dtype=dtype)

    return d
