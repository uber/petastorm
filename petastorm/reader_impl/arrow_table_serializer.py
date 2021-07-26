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

import pyarrow as pa


class ArrowTableSerializer(object):
    """This implementation of serializer is used to facilitate faster serialization of pyarrow tables.
    Even a better solution would be to use shared memory (e.g. using plasma)."""

    def serialize(self, rows):
        # Need to be able to serialize `None`. A bit hacky, but we use an empty buffer to encode 'None'.
        sink = pa.BufferOutputStream()
        writer = pa.RecordBatchStreamWriter(sink, rows.schema)
        writer.write_table(rows)
        writer.close()
        return sink.getvalue()

    def deserialize(self, serialized_rows):
        reader = pa.ipc.open_stream(serialized_rows)
        table = reader.read_all()
        return table
