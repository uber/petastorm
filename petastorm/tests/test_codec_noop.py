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
from decimal import Decimal

import numpy as np
from pyspark.sql.types import StringType, ArrayType

from petastorm.codecs import NoopCodec
from petastorm.unischema import UnischemaField


def test_nested_value():
    codec = NoopCodec(ArrayType(ArrayType(StringType())))
    field = UnischemaField(name='field_string', numpy_dtype=np.string_, shape=(None, None), codec=codec, nullable=False)
    nested_array = [['a', 'b'], ['c'], ['d']]
    assert codec.decode(field, codec.encode(field, nested_array)) == nested_array
