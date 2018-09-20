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
from time import sleep

import numpy as np
import six

try:
    from unittest import mock
except ImportError:
    from mock import mock

from petastorm.benchmark.throughput import reader_throughput, reader_v2_throughput, \
    _filter_schema_fields, _time_warmup_and_work, WorkerPoolType, ReadMethod
from petastorm.unischema import Unischema, UnischemaField


def test_pure_python_process_pool_run(synthetic_dataset):
    reader_throughput(synthetic_dataset.url, ['id'], warmup_cycles_count=5, measure_cycles_count=5,
                      pool_type=WorkerPoolType.PROCESS, loaders_count=1, read_method=ReadMethod.PYTHON,
                      spawn_new_process=False)


def test_tf_thread_pool_run(synthetic_dataset):
    reader_throughput(synthetic_dataset.url, ['id', 'id2'], warmup_cycles_count=5, measure_cycles_count=5,
                      pool_type=WorkerPoolType.THREAD, loaders_count=1, read_method=ReadMethod.TF)


def test_pure_python_thread_pool_run(synthetic_dataset):
    # Use a regex to match field name ('i.' instead of 'id')
    reader_throughput(synthetic_dataset.url, ['i.'], warmup_cycles_count=5, measure_cycles_count=5,
                      pool_type=WorkerPoolType.THREAD, loaders_count=1, read_method=ReadMethod.PYTHON)


def test_pure_python_dummy_pool_run(synthetic_dataset):
    # Use a regex to match field name ('i.' instead of 'id')
    reader_throughput(synthetic_dataset.url, ['i.'], warmup_cycles_count=5, measure_cycles_count=5,
                      pool_type=WorkerPoolType.NONE, loaders_count=1, read_method=ReadMethod.PYTHON)


def test_all_fields(synthetic_dataset):
    reader_throughput(synthetic_dataset.url, None, warmup_cycles_count=5, measure_cycles_count=5,
                      pool_type=WorkerPoolType.THREAD, loaders_count=1, read_method=ReadMethod.PYTHON)


def test_experimental_reader(synthetic_dataset):
    reader_v2_throughput(synthetic_dataset.url, None, warmup_cycles_count=5, measure_cycles_count=5,
                         pool_type=WorkerPoolType.THREAD, loaders_count=1, read_method=ReadMethod.PYTHON)


def test_tf_thread_pool_run_experimental(synthetic_dataset):
    reader_v2_throughput(synthetic_dataset.url, field_regex=[r'\bid\b', r'\bmatrix\b'], warmup_cycles_count=5,
                         measure_cycles_count=5, pool_type=WorkerPoolType.THREAD, loaders_count=1,
                         read_method=ReadMethod.TF)


def test_filter_schema_fields_from_url():
    TestSchema = Unischema('TestSchema', [
        UnischemaField('int32', np.int32, (), None, False),
        UnischemaField('uint8', np.uint8, (), None, False),
        UnischemaField('uint16', np.uint16, (), None, False),
    ])

    assert _filter_schema_fields(TestSchema, ['.*nt.*6']) == [TestSchema.uint16]
    assert _filter_schema_fields(TestSchema, ['nomatch']) == []
    assert _filter_schema_fields(TestSchema, ['.*']) == list(TestSchema.fields.values())
    assert _filter_schema_fields(TestSchema, ['int32', 'uint8']) == [TestSchema.int32, TestSchema.uint8]


def test_run_benchmark_cycle_length_of_warmup_and_measure_cycles():
    measurable = mock.Mock()
    reader_mock = mock.Mock()
    _time_warmup_and_work(reader_mock, 2, 3, measurable.next_item)
    assert 5 == measurable.next_item.call_count

    measurable = mock.Mock()
    _time_warmup_and_work(reader_mock, 6, 7, measurable.next_item)
    assert 13 == measurable.next_item.call_count


def test_time_measure():
    T = 1.2
    measurable = mock.Mock()
    reader = mock.Mock()
    reader.diagnostics.side_effect = {'some_diags': 1}
    wait_times = [0.0, T, T]

    def mock_next_item():
        a = 1
        for _ in six.moves.xrange(10000):
            a += 1
        sleep(wait_times.pop(0))
        return 0

    measurable.next_item.side_effect = mock_next_item
    result = _time_warmup_and_work(reader, 1, 2, measurable.next_item)
    assert result.time_mean >= T / 2.0
    assert result.samples_per_second < 1.0 / T
    assert result.memory_info
