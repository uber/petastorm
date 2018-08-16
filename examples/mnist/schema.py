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
from pyspark.sql.types import IntegerType

from petastorm.codecs import ScalarCodec, NdarrayCodec
from petastorm.unischema import Unischema, UnischemaField

MnistSchema = Unischema('MnistSchema', [
    UnischemaField('idx', np.int_, (), ScalarCodec(IntegerType()), False),
    UnischemaField('digit', np.int_, (), ScalarCodec(IntegerType()), False),
    UnischemaField('image', np.uint8, (28, 28), NdarrayCodec(), False),
])
