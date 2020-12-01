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
import pandas as pd
import pyarrow as pa
from pandas.util.testing import assert_frame_equal

from petastorm.reader_impl.arrow_table_serializer import ArrowTableSerializer


def test_random_table():
    """Serialize/deserialize some small table"""
    expected_dataframe = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))
    expected_table = pa.Table.from_pandas(expected_dataframe)

    serializer = ArrowTableSerializer()
    actual_table = serializer.deserialize(serializer.serialize(expected_table))
    assert_frame_equal(actual_table.to_pandas(), expected_dataframe)


def test_empty_table():
    """See that we can transmit empty tables"""
    expected_dataframe = pd.DataFrame(np.empty(shape=(0, 4), dtype=np.int8), columns=list('ABCD'))
    expected_table = pa.Table.from_pandas(expected_dataframe)

    serializer = ArrowTableSerializer()
    stream = serializer.serialize(expected_table)
    actual_table = serializer.deserialize(stream)
    assert_frame_equal(actual_table.to_pandas(), expected_dataframe)
