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
import pyarrow as pa

from petastorm.pyarrow_helpers.batching_table_queue import BatchingTableQueue


def _new_record_batch(values):
    sequence = pa.array(values)
    return pa.RecordBatch.from_arrays([sequence, sequence], ['a', 'b'])


def test_single_table_of_10_rows_added_and_2_batches_of_4_read():
    """Add a single table composed of a single batch with 0..9 rows into batcher. Then read two batches of 4
    and verify that no more batches can be read"""

    # Table with two columns. Each column with 0..9 sequence
    one_batch_of_10_records = [_new_record_batch(range(0, 10))]
    table_0_10 = pa.Table.from_batches(one_batch_of_10_records)

    batcher = BatchingTableQueue(4)
    assert batcher.empty()

    # Load 10 rows into batcher
    batcher.put(table_0_10)

    # Get first batch of 4
    assert not batcher.empty()
    next_batch = batcher.get()

    assert 4 == next_batch.num_rows
    np.testing.assert_equal(next_batch.column(0).to_pylist(), list(range(0, 4)))
    np.testing.assert_equal(next_batch.column(1).to_pylist(), list(range(0, 4)))

    # Get second batch of 4
    assert not batcher.empty()
    next_batch = batcher.get()

    assert 4 == next_batch.num_rows
    np.testing.assert_equal(next_batch.column(0).to_pylist(), list(range(4, 8)))
    np.testing.assert_equal(next_batch.column(1).to_pylist(), list(range(4, 8)))

    # No more batches available
    assert batcher.empty()


def test_two_tables_of_10_added_reading_5_batches_of_4():
    """Add two tables to batcher and read a batch that covers parts of both tables"""
    table_0_9 = pa.Table.from_batches([_new_record_batch(range(0, 10))])
    table_10_19 = pa.Table.from_batches([_new_record_batch(range(10, 20))])

    batcher = BatchingTableQueue(4)
    assert batcher.empty()

    batcher.put(table_0_9)
    batcher.put(table_10_19)

    for i in range(5):
        assert not batcher.empty()
        next_batch = batcher.get()

        assert (i != 4) == (not batcher.empty())

        assert 4 == next_batch.num_rows
        expected_values = list(range(i * 4, i * 4 + 4))
        np.testing.assert_equal(next_batch.column(0).to_pylist(), expected_values)
        np.testing.assert_equal(next_batch.column(1).to_pylist(), expected_values)


def test_read_batches_larger_than_a_table_added():
    """Add a single table composed of 10 one row ten_batches_each_with_one_record. Then read-out two batches of 4
    and verify that no more batches can be read"""
    ten_batches_each_with_one_record = [_new_record_batch([i]) for i in range(10)]
    table_0_10 = pa.Table.from_batches(ten_batches_each_with_one_record)

    batcher = BatchingTableQueue(4)
    batcher.put(table_0_10)

    assert not batcher.empty()
    next_batch = batcher.get()

    assert 4 == next_batch.num_rows
    np.testing.assert_equal(next_batch.column(0).to_pylist(), list(range(0, 4)))
    np.testing.assert_equal(next_batch.column(1).to_pylist(), list(range(0, 4)))

    assert not batcher.empty()
    next_batch = batcher.get()

    assert 4 == next_batch.num_rows
    np.testing.assert_equal(next_batch.column(0).to_pylist(), list(range(4, 8)))
    np.testing.assert_equal(next_batch.column(1).to_pylist(), list(range(4, 8)))

    assert batcher.empty()


def test_batch_size_of_one():
    """Try if BatchingTableQueue can be used to retrieve row-by-row data (batch size of 1)"""
    batches = [_new_record_batch([i]) for i in range(3)]
    table = pa.Table.from_batches(batches)

    batcher = BatchingTableQueue(1)
    batcher.put(table)

    for _ in range(3):
        assert not batcher.empty()
        next_batch = batcher.get()
        assert 1 == next_batch.num_rows

    assert batcher.empty()


def test_random_table_size_and_random_batch_sizes():
    """Add a random number of rows, then read a random number of batches. Repeat multiple times."""
    batch_size = 5
    input_table_size = 50
    read_iter_count = 1000

    batcher = BatchingTableQueue(batch_size)
    write_seq = 0
    read_seq = 0

    for _ in range(read_iter_count):
        next_batch_size = np.random.randint(0, input_table_size)
        new_batch = _new_record_batch(list(range(write_seq, write_seq + next_batch_size)))
        write_seq += next_batch_size

        batcher.put(pa.Table.from_batches([new_batch]))

        next_read = np.random.randint(1, input_table_size // batch_size)
        for _ in range(next_read):
            if not batcher.empty():
                read_batch = batcher.get()
                for value in read_batch.columns[0]:
                    assert value.as_py() == read_seq
                    read_seq += 1

    assert read_seq > 0
