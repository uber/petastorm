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

from __future__ import division

import numpy as np
import pytest
import six

from petastorm.predicates import in_lambda
from petastorm.reader import Reader
from petastorm.weighted_sampling_reader import WeightedSamplingReader
from petastorm.test_util.reader_mock import ReaderMock
from petastorm.unischema import Unischema, UnischemaField
from petastorm.workers_pool.dummy_pool import DummyPool

TestSchema = Unischema('TestSchema', [
    UnischemaField('f1', np.int32, (), None, False),
])

reader0 = ReaderMock(TestSchema, lambda _: {'f1': 0})
reader1 = ReaderMock(TestSchema, lambda _: {'f1': 1})
reader2 = ReaderMock(TestSchema, lambda _: {'f1': 2})


def _count_mixed(readers, probabilities, num_of_reads):
    result = len(probabilities) * [0]

    with WeightedSamplingReader(readers, probabilities) as mixer:
        for _ in six.moves.xrange(num_of_reads):
            reader_index = next(mixer).f1
            result[reader_index] += 1

    return result


def test_select_only_one_of_readers():
    num_of_reads = 1000
    np.testing.assert_array_equal(_count_mixed([reader0, reader1], [1.0, 0.0], num_of_reads), [num_of_reads, 0])
    np.testing.assert_array_equal(_count_mixed([reader0, reader1], [0.0, 1.0], num_of_reads), [0, num_of_reads])

    np.testing.assert_array_equal(
        _count_mixed([reader0, reader1, reader2], [0.0, 1.0, 0.0], num_of_reads), [0, num_of_reads, 0])
    np.testing.assert_array_equal(
        _count_mixed([reader0, reader1, reader2], [0.0, 0.0, 1.0], num_of_reads), [0, 0, num_of_reads])


def test_not_normalized_probabilities():
    num_of_reads = 1000
    mix_10_90 = _count_mixed([reader0, reader1], [10, 90], num_of_reads)
    np.testing.assert_allclose(mix_10_90, [num_of_reads * 0.1, num_of_reads * 0.9], atol=num_of_reads / 10)


def test_mixing():
    num_of_reads = 1000
    mix_10_90 = _count_mixed([reader0, reader1], [0.1, 0.9], num_of_reads)

    np.testing.assert_allclose(mix_10_90, [num_of_reads * 0.1, num_of_reads * 0.9], atol=num_of_reads / 10)

    mix_10_50_40 = _count_mixed([reader0, reader1, reader2], [0.1, 0.5, 0.4], num_of_reads)
    np.testing.assert_allclose(mix_10_50_40, [num_of_reads * 0.1, num_of_reads * 0.5, num_of_reads * 0.4],
                               atol=num_of_reads / 10)


def test_real_reader(synthetic_dataset):
    readers = [Reader(synthetic_dataset.url, predicate=in_lambda(['id'], lambda id: id % 2 == 0), num_epochs=None,
                      reader_pool=DummyPool()),
               Reader(synthetic_dataset.url, predicate=in_lambda(['id'], lambda id: id % 2 == 1), num_epochs=None,
                      reader_pool=DummyPool())]
    results = [0, 0]
    num_of_reads = 300
    with WeightedSamplingReader(readers, [0.5, 0.5]) as mixer:
        # Piggyback on this test to verify container interface of the WeightedSamplingReader
        for i, sample in enumerate(mixer):
            next_id = sample.id % 2
            results[next_id] += 1
            if i >= num_of_reads:
                break

    np.testing.assert_allclose(results, [num_of_reads * 0.5, num_of_reads * 0.5], atol=num_of_reads / 10)


def test_bad_arguments():
    with pytest.raises(ValueError):
        WeightedSamplingReader([reader1], [0.1, 0.9])
