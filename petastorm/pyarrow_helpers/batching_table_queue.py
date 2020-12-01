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

from collections import deque

import pyarrow as pa


class BatchingTableQueue(object):
    def __init__(self, batch_size):
        """The class is a FIFO queue. Arrow tables are added to the queue. When read, rows are regrouped into Arrow
        tables of a fixed size specified during construction of the object. The order of the rows in the output tables
        is the same as the order of the rows in the input tables.

        :param batch_size: number of rows in tables that will be returned by the ``get`` method.
        """
        self._batch_size = batch_size
        self._buffer = deque()
        self._head_idx = 0
        self._cumulative_len = 0

    def put(self, table):
        """Adds a table to the queue. All tables added during lifetime of an instance must have the same schema.

        :param table: An instance of a pyarrow table.
        :return: None
        """

        # We store a list of arrow batches. When retrieving, we consume parts or entire batches, until batch_size of
        # rows are acquired.
        record_batches = table.to_batches()
        for record_batch in record_batches:
            self._buffer.append(record_batch)
            self._cumulative_len += record_batch.num_rows

    def empty(self):
        """Checks if more tables can be returned by get. If the number of rows in the internal buffer is less then
        ``batch_size``, empty would return False.
        """
        return self._head_idx + self._batch_size > self._cumulative_len

    def get(self):
        """Return a table with ``batch_size`` number of rows.

        :return: An instance of an Arrow table with exactly ``batch_size`` rows.
        """

        assert not self.empty()

        # head_idx points to the next row in the buffer[0] batch to be consumed.
        # Accumulate selices/full batches until result_rows reaches desired batch_size.

        # Pop left of the deque once exhausted all rows there.
        result = []
        result_rows = 0
        while result_rows < self._batch_size and self._cumulative_len > 0:
            head = self._buffer[0]
            piece = head[self._head_idx:self._head_idx + self._batch_size - result_rows]
            self._head_idx += piece.num_rows
            result_rows += piece.num_rows
            result.append(piece)

            if head.num_rows == self._head_idx:
                self._head_idx = 0
                self._buffer.popleft()
                self._cumulative_len -= head.num_rows

        return pa.Table.from_batches(result)
