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

import functools
import operator
from copy import copy

import numpy as np
import pytest
import tensorflow.compat.v1 as tf  # pylint: disable=import-error

from petastorm import make_reader, make_batch_reader
from petastorm.ngram import NGram
from petastorm.predicates import in_lambda
from petastorm.tests.test_common import TestSchema
from petastorm.tests.test_tf_utils import create_tf_graph
from petastorm.tf_utils import make_petastorm_dataset

_EXCLUDE_FIELDS = set(TestSchema.fields.values()) \
                  - {TestSchema.matrix_nullable, TestSchema.string_array_nullable, TestSchema.decimal,
                     TestSchema.integer_nullable}

MINIMAL_READER_FLAVOR_FACTORIES = [
    lambda url, **kwargs: make_reader(url, **_merge_params({'reader_pool_type': 'dummy',
                                                            'schema_fields': _EXCLUDE_FIELDS}, kwargs)),
]

ALL_READER_FLAVOR_FACTORIES = MINIMAL_READER_FLAVOR_FACTORIES + [
    lambda url, **kwargs: make_reader(url, **_merge_params({'reader_pool_type': 'thread', 'workers_count': 1,
                                                            'schema_fields': _EXCLUDE_FIELDS}, kwargs)),
    lambda url, **kwargs: make_reader(url, **_merge_params({'reader_pool_type': 'process', 'workers_count': 1,
                                                            'schema_fields': _EXCLUDE_FIELDS}, kwargs)),
]


def _merge_params(base, overwrite):
    """Merges two dictionaries when values from ``overwrite`` takes precedence over values of ``base`` dictionary.

    Both input parameters are not modified.

    :param base: A dictionary
    :param overwrite: A dictionary. If a value with the same key exists in ``base``, it is overwritten by the value from
      this dictionary.
    :return: A combined dictionary
    """
    # Create a shallow copy of base
    combined = copy(base)
    combined.update(overwrite)
    return combined


@pytest.mark.forked
@pytest.mark.parametrize('reader_factory', ALL_READER_FLAVOR_FACTORIES)
@create_tf_graph
def test_with_one_shot_iterator(synthetic_dataset, reader_factory):
    """Just a bunch of read and compares of all values to the expected values"""
    with reader_factory(synthetic_dataset.url) as reader:
        dataset = make_petastorm_dataset(reader)
        iterator = dataset.make_one_shot_iterator()

        # Make sure we have static shape info for all fields
        for shape in dataset.output_shapes:
            # TODO(yevgeni): check that the shapes are actually correct, not just not None
            assert shape.dims is not None

        # Read a bunch of entries from the dataset and compare the data to reference
        with tf.Session() as sess:
            iterator = iterator.get_next()
            for _, _ in enumerate(synthetic_dataset.data):
                actual = sess.run(iterator)._asdict()
                expected = next(d for d in synthetic_dataset.data if d['id'] == actual['id'])
                for key in actual.keys():
                    if isinstance(expected[key], str):
                        # Tensorflow returns all strings as bytes in python3. So we will need to decode it
                        actual_value = actual[key].decode()
                    elif isinstance(expected[key], np.ndarray) and expected[key].dtype.type == np.unicode_:
                        actual_value = np.array([item.decode() for item in actual[key]])
                    else:
                        actual_value = actual[key]

                    np.testing.assert_equal(actual_value, expected[key])

            # Exhausted one full epoch. Fetching next value should trigger OutOfRangeError
            with pytest.raises(tf.errors.OutOfRangeError):
                sess.run(iterator)


@pytest.mark.forked
@pytest.mark.parametrize('reader_factory', ALL_READER_FLAVOR_FACTORIES)
@create_tf_graph
def test_with_dataset_repeat(synthetic_dataset, reader_factory):
    """``tf.data.Dataset``'s ``repeat`` is not recommended to used on ``make_petastorm_dataset`` due to high costs of
    ``Reader initialization``. A user should use ``Reader`` built-in epochs support, or use
    ``tf.data.Dataset``'s ``cache``. Check that we trigger a warning to alert of misuse."""
    with reader_factory(synthetic_dataset.url) as reader:
        dataset = make_petastorm_dataset(reader)

        dataset = dataset.repeat(2)

        iterator = dataset.make_one_shot_iterator()

        # Read a bunch of entries from the dataset and compare the data to reference
        with tf.Session() as sess:
            iterator = iterator.get_next()

            for _, _ in enumerate(synthetic_dataset.data):
                sess.run(iterator)

            match_str = 'Running multiple iterations over make_petastorm_dataset is not recommend for performance issue'
            with pytest.warns(UserWarning, match=match_str):
                sess.run(iterator)


@pytest.mark.forked
@pytest.mark.parametrize('reader_factory', ALL_READER_FLAVOR_FACTORIES)
@create_tf_graph
def test_with_dataset_repeat_after_cache(synthetic_dataset, reader_factory):
    """ Check if ``tf.data.Dataset``'s ``repeat`` works after ``tf.data.Dataset``'s ``cache``."""
    epochs = 3
    with reader_factory(synthetic_dataset.url, schema_fields=[TestSchema.id]) as reader:
        dataset = make_petastorm_dataset(reader)
        dataset = dataset.cache()
        dataset = dataset.repeat(epochs)
        iterator = dataset.make_one_shot_iterator()
        it_op = iterator.get_next()
        # Check if dataset generates same result in every epoch.
        with tf.Session() as sess:
            with pytest.warns(None):
                # Expect no warnings since cache() is called before repeat()
                for _ in range(epochs):
                    actual_res = []
                    for _, _ in enumerate(synthetic_dataset.data):
                        actual = sess.run(it_op)._asdict()
                        actual_res.append(actual["id"])
                    expected_res = list(range(len(synthetic_dataset.data)))
                    # sort dataset output since row_groups are shuffled from reader.
                    np.testing.assert_equal(sorted(actual_res), expected_res)

            # Exhausted all epochs. Fetching next value should trigger OutOfRangeError
            with pytest.raises(tf.errors.OutOfRangeError):
                sess.run(it_op)


@pytest.mark.forked
@pytest.mark.parametrize('reader_factory', ALL_READER_FLAVOR_FACTORIES)
@create_tf_graph
def test_some_processing_functions(synthetic_dataset, reader_factory):
    """Try several ``tf.data.Dataset`` dataset operations on make_petastorm_dataset"""

    # reader1 will have a single row with id=1, reader2: a single row with id=2

    # Using functools.partial(_eq, 1)) which is equivalent to lambda x: x==1 because standard python pickle
    # can not pickle this lambda
    with reader_factory(synthetic_dataset.url,
                        predicate=in_lambda(['id'], functools.partial(operator.eq, 1))) as reader1:
        with reader_factory(synthetic_dataset.url,
                            predicate=in_lambda(['id'], functools.partial(operator.eq, 2))) as reader2:
            dataset = make_petastorm_dataset(reader1) \
                .prefetch(10) \
                .concatenate(make_petastorm_dataset(reader2)) \
                .map(lambda x: x.id) \
                .batch(2)

            next_sample = dataset.make_one_shot_iterator().get_next()

            with tf.Session() as sess:
                # 'actual' is expected to be content of id column of a concatenated dataset
                actual = sess.run(next_sample)
                np.testing.assert_array_equal(actual, [1, 2])


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
@create_tf_graph
def test_dataset_with_ngrams(synthetic_dataset, reader_factory):
    ngram = NGram({-1: [TestSchema.id, TestSchema.image_png], 2: [TestSchema.id]}, 100, TestSchema.id)
    with reader_factory(synthetic_dataset.url, schema_fields=ngram, num_epochs=1) as reader:
        dataset = make_petastorm_dataset(reader)
        next_sample = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            while True:
                try:
                    actual = sess.run(next_sample)
                    assert actual[-1].id + 3 == actual[2].id
                    assert np.all(actual[-1].image_png.shape > (10, 10, 3))
                except tf.errors.OutOfRangeError:
                    break


@pytest.mark.forked
@create_tf_graph
def test_non_petastorm_with_many_colums_with_one_shot_iterator(many_columns_non_petastorm_dataset):
    """Just a bunch of read and compares of all values to the expected values"""
    with make_batch_reader(many_columns_non_petastorm_dataset.url, workers_count=1) as reader:
        dataset = make_petastorm_dataset(reader)
        iterator = dataset.make_one_shot_iterator()

        # Make sure we have static shape info for all fields
        for shape in dataset.output_shapes:
            # TODO(yevgeni): check that the shapes are actually correct, not just not None
            assert shape.dims is not None

        # Read a bunch of entries from the dataset and compare the data to reference
        with tf.Session() as sess:
            get_next = iterator.get_next()
            sample = sess.run(get_next)._asdict()
            assert set(sample.keys()) == set(many_columns_non_petastorm_dataset.data[0].keys())


@pytest.mark.forked
@create_tf_graph
def test_non_petastorm_with_many_colums_epoch_count(many_columns_non_petastorm_dataset):
    """Just a bunch of read and compares of all values to the expected values"""
    expected_num_epochs = 4
    with make_batch_reader(many_columns_non_petastorm_dataset.url, workers_count=1,
                           num_epochs=expected_num_epochs) as reader:
        dataset = make_petastorm_dataset(reader)
        iterator = dataset.make_one_shot_iterator()

        # Read a bunch of entries from the dataset and compare the data to reference
        with tf.Session() as sess:
            get_next = iterator.get_next()
            rows_count = 0
            while True:
                try:
                    sample = sess.run(get_next)._asdict()
                    rows_count += sample['col_0'].shape[0]
                except tf.errors.OutOfRangeError:
                    break

            assert expected_num_epochs * len(many_columns_non_petastorm_dataset.data) == rows_count
