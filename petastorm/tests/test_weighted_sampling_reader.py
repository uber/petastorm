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
import tensorflow.compat.v1 as tf  # pylint: disable=import-error

from petastorm import make_reader
from petastorm.ngram import NGram
from petastorm.predicates import in_lambda
from petastorm.pytorch import DataLoader
from petastorm.test_util.reader_mock import ReaderMock
from petastorm.tf_utils import tf_tensors, make_petastorm_dataset
from petastorm.unischema import Unischema, UnischemaField
from petastorm.weighted_sampling_reader import WeightedSamplingReader
from petastorm.tests.test_tf_utils import create_tf_graph

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
    readers = [make_reader(synthetic_dataset.url, predicate=in_lambda(['id'], lambda id: id % 2 == 0), num_epochs=None,
                           reader_pool_type='dummy'),
               make_reader(synthetic_dataset.url, predicate=in_lambda(['id'], lambda id: id % 2 == 1), num_epochs=None,
                           reader_pool_type='dummy')]
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


@create_tf_graph
def test_with_tf_tensors(synthetic_dataset):
    fields_to_read = ['id.*', 'image_png']
    readers = [make_reader(synthetic_dataset.url, schema_fields=fields_to_read, workers_count=1),
               make_reader(synthetic_dataset.url, schema_fields=fields_to_read, workers_count=1)]

    with WeightedSamplingReader(readers, [0.5, 0.5]) as mixer:
        mixed_tensors = tf_tensors(mixer)

        with tf.Session() as sess:
            sess.run(mixed_tensors)


def test_schema_mismatch(synthetic_dataset):
    readers = [make_reader(synthetic_dataset.url, schema_fields=['id'], workers_count=1),
               make_reader(synthetic_dataset.url, schema_fields=['image_png'], workers_count=1)]

    with pytest.raises(ValueError, match='.*should have the same schema.*'):
        WeightedSamplingReader(readers, [0.5, 0.5])


@create_tf_graph
def test_ngram_mix(synthetic_dataset):
    ngram1_fields = {
        -1: ['id', ],
        0: ['id', 'image_png'],
    }

    ts_field = '^id$'

    ngram1 = NGram(fields=ngram1_fields, delta_threshold=10, timestamp_field=ts_field)
    ngram2 = NGram(fields=ngram1_fields, delta_threshold=10, timestamp_field=ts_field)

    readers = [make_reader(synthetic_dataset.url, schema_fields=ngram1, workers_count=1),
               make_reader(synthetic_dataset.url, schema_fields=ngram2, workers_count=1)]

    with WeightedSamplingReader(readers, [0.5, 0.5]) as mixer:
        mixed_tensors = tf_tensors(mixer)

        with tf.Session() as sess:
            for _ in range(10):
                actual = sess.run(mixed_tensors)
                assert set(actual.keys()) == {-1, 0}


def test_ngram_mismsatch(synthetic_dataset):
    ngram1_fields = {
        -1: ['id', ],
        0: ['id', 'image_png'],
    }

    ngram2_fields = {
        -1: ['id', 'image_png'],
        0: ['id', ],
    }

    ts_field = '^id$'

    ngram1 = NGram(fields=ngram1_fields, delta_threshold=10, timestamp_field=ts_field)
    ngram2 = NGram(fields=ngram2_fields, delta_threshold=10, timestamp_field=ts_field)

    readers = [make_reader(synthetic_dataset.url, schema_fields=ngram1, workers_count=1),
               make_reader(synthetic_dataset.url, schema_fields=ngram2, workers_count=1)]

    with pytest.raises(ValueError, match='.*ngram.*'):
        WeightedSamplingReader(readers, [0.5, 0.5])


@create_tf_graph
def test_with_tf_data_api(synthetic_dataset):
    """Verify that WeightedSamplingReader is compatible with make_petastorm_dataset"""

    np.random.seed(42)

    fields_to_read = ['id.*', 'image_png']

    # Use cur_shard=0, shard_count=2 to get only half samples from the second reader.
    readers = [make_reader(synthetic_dataset.url, schema_fields=fields_to_read, workers_count=1),
               make_reader(synthetic_dataset.url, schema_fields=fields_to_read, workers_count=1,
                           cur_shard=0, shard_count=2)]

    with WeightedSamplingReader(readers, [0.5, 0.5]) as mixer:
        dataset = make_petastorm_dataset(mixer)
        iterator = dataset.make_one_shot_iterator()
        tensor = iterator.get_next()
        rows_count = 0
        with tf.Session() as sess:
            while True:
                try:
                    sess.run(tensor)
                    rows_count += 1
                except tf.errors.OutOfRangeError:
                    break

        # We expect iterations to finish once the second read has exhausted its samples. For each sample in the
        # second reaader we read approximately 1 sample from the first.
        expected_rows_approx = len(synthetic_dataset.data)
        np.testing.assert_allclose(rows_count, expected_rows_approx, atol=20)


def test_with_torch_api(synthetic_dataset):
    """Verify that WeightedSamplingReader is compatible with petastorm.pytorch.DataLoader"""
    readers = [reader0, reader1]

    with WeightedSamplingReader(readers, [0.5, 0.5]) as mixer:
        assert not mixer.batched_output
        sample = next(mixer)
        assert sample is not None
        with DataLoader(mixer, batch_size=2) as loader:
            for batch in loader:
                assert batch['f1'].shape[0] == 2
                break
