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

import pyarrow

from petastorm.codecs import decimal_to_str


class PyArrowSerializer(object):

    def serialize(self, rows):
        # pyarrow.serialize does not support decimals.
        decimal_to_str(rows)
        return pyarrow.serialize(rows).to_buffer()

    def deserialize(self, serialized_rows):
        return pyarrow.read_serialized(serialized_rows).deserialize()
