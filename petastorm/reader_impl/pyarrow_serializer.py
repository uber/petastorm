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

import pyarrow
from pyarrow import register_default_serialization_handlers


class PyArrowSerializer(object):

    def serialize(self, rows):
        return pyarrow.serialize(rows, self._get_serialization_context()).to_buffer()

    def deserialize(self, serialized_rows):
        return pyarrow.deserialize(serialized_rows, self._get_serialization_context())

    def __getstate__(self):
        state = self.__dict__.copy()
        # The context is not picklable, so we have to delete it from the state when saving
        # we initialize and create it lazily in _get_serialization_context
        if '_context' in state:
            del state['_context']
        return state

    def _get_serialization_context(self):
        # Create _context lazily.
        if not hasattr(self, '_context'):
            self._context = pyarrow.SerializationContext()
            register_default_serialization_handlers(self._context)
            self._context.register_type(Decimal, 'decimal.Decimal', pickle=True)

        return self._context
