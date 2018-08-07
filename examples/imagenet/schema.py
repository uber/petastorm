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
from pyspark.sql.types import StringType

from petastorm.codecs import ScalarCodec, CompressedImageCodec
from petastorm.unischema import Unischema, UnischemaField

ImagenetSchema = Unischema('ImagenetSchema', [
    UnischemaField('noun_id', np.string_, (), ScalarCodec(StringType()), False),
    UnischemaField('text', np.string_, (), ScalarCodec(StringType()), False),
    UnischemaField('image', np.uint8, (None, None, 3), CompressedImageCodec('png'), False),
])
