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
import operator
import os
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor
from shutil import rmtree, copytree
from six.moves.urllib.parse import urlparse

import numpy as np
import pyarrow.hdfs
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, ShortType, StringType

from unittest import mock

from petastorm import make_reader, make_batch_reader, TransformSpec
from petastorm.codecs import ScalarCodec, CompressedImageCodec
from petastorm.errors import NoDataAvailableError
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.predicates import in_lambda
from petastorm.selectors import SingleIndexSelector, IntersectIndexSelector, UnionIndexSelector
from petastorm.tests.test_common import create_test_dataset, TestSchema
from petastorm.tests.test_end_to_end_predicates_impl import \
    PartitionKeyInSetPredicate, EqualPredicate, VectorizedEqualPredicate
from petastorm.unischema import UnischemaField, Unischema


# pylint: disable=unnecessary-lambda
MINIMAL_READER_FLAVOR_FACTORIES = [
    lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs),
]

# pylint: disable=unnecessary-lambda
ALL_READER_FLAVOR_FACTORIES = MINIMAL_READER_FLAVOR_FACTORIES + [
    lambda url, **kwargs: make_reader(url, reader_pool_type='thread', **kwargs),
    lambda url, **kwargs: make_reader(url, reader_pool_type='process', workers_count=2, **kwargs),
    lambda url, **kwargs: make_reader(url, reader_pool_type='process', workers_count=2, **kwargs),
]

SCALAR_FIELDS = [f for f in TestSchema.fields.values() if isinstance(f.codec, ScalarCodec)]

SCALAR_ONLY_READER_FACTORIES = [
    lambda url, **kwargs: make_batch_reader(url, reader_pool_type='dummy', **kwargs),
    lambda url, **kwargs: make_batch_reader(url, reader_pool_type='process', workers_count=2, **kwargs),
]


def _check_simple_reader(reader, expected_data, expected_rows_count=None, check_types=True, limit_checked_rows=None):
    # Read a bunch of entries from the dataset and compare the data to reference
    def _type(v):
        if isinstance(v, np.ndarray):
            if v.dtype.str.startswith('|S'):
                return '|S'
            else:
                return v.dtype
        else:
            return type(v)

    expected_rows_count = expected_rows_count or len(expected_data)
    count = 0

    for i, row in enumerate(reader):
        if limit_checked_rows and i >= limit_checked_rows:
            break

        actual = row._asdict()
        expected = next(d for d in expected_data if d['id'] == actual['id'])
        np.testing.assert_equal(actual, expected)
        actual_types = {k: _type(v) for k, v in actual.items()}
        expected_types = {k: _type(v) for k, v in expected.items()}
        assert not check_types or actual_types == expected_types
        count += 1

    if limit_checked_rows:
        assert count == min(expected_rows_count, limit_checked_rows)
    else:
        assert count == expected_rows_count


def _readout_all_ids(reader, limit=None):
    ids = []
    for i, row in enumerate(reader):
        if limit is not None and i >= limit:
            break
        ids.append(row.id)

    # Flatten ids if reader returns batches (make_batch_reader)
    if isinstance(ids[0], np.ndarray):
        ids = [i for arr in ids for i in arr]

    return ids


@pytest.mark.parametrize('reader_factory', ALL_READER_FLAVOR_FACTORIES)
def test_simple_read(synthetic_dataset, reader_factory):
    """Just a bunch of read and compares of all values to the expected values using the different reader pools"""
    with reader_factory(synthetic_dataset.url) as reader:
        _check_simple_reader(reader, synthetic_dataset.data)


@pytest.mark.parametrize('reader_factory', [
    lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs)
])
def test_transform_function(synthetic_dataset, reader_factory):
    """"""

    def double_matrix(sample):
        sample['matrix'] *= 2
        return sample

    with reader_factory(synthetic_dataset.url, schema_fields=[TestSchema.id, TestSchema.matrix],
                        transform_spec=TransformSpec(double_matrix)) as reader:
        actual = next(reader)
        original_sample = next(d for d in synthetic_dataset.data if d['id'] == actual.id)
        expected_matrix = original_sample['matrix'] * 2
        np.testing.assert_equal(expected_matrix, actual.matrix)


@pytest.mark.parametrize('reader_factory', [
    lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs)
])
def test_transform_function_returns_a_new_dict(synthetic_dataset, reader_factory):
    """"""

    def double_matrix(sample):
        return {'id': -1}

    with reader_factory(synthetic_dataset.url, schema_fields=[TestSchema.id],
                        transform_spec=TransformSpec(double_matrix)) as reader:
        all_samples = list(reader)
        actual_ids = list(map(lambda x: x.id, all_samples))

        np.testing.assert_equal(actual_ids, [-1] * len(synthetic_dataset.data))


@pytest.mark.parametrize('reader_factory', [
    lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs)
])
def test_transform_remove_field(synthetic_dataset, reader_factory):
    """Make sure we apply transform only after we apply the predicate"""

    with reader_factory(synthetic_dataset.url, schema_fields=[TestSchema.id, TestSchema.id2],
                        transform_spec=TransformSpec(removed_fields=['id2'])) as reader:
        row = next(reader)
        assert 'id2' not in row._fields
        assert 'id' in row._fields


@pytest.mark.parametrize('reader_factory', [
    lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs)
])
def test_transform_function_with_predicate(synthetic_dataset, reader_factory):
    """Make sure we apply transform only after we apply the predicate"""

    with reader_factory(synthetic_dataset.url, schema_fields=[TestSchema.id, TestSchema.id2],
                        predicate=in_lambda(['id2'], lambda id2: id2 == 1),
                        transform_spec=TransformSpec(removed_fields=['id2'])) as reader:
        rows = list(reader)
        assert 'id2' not in rows[0]._fields
        actual_ids = np.asarray(list(row.id for row in rows))
        assert actual_ids.size > 0
        # In the test data id2 = id % 2, which means we expect only odd ids to remain after
        # we apply lambda id2: id2 == 1 predicate.
        assert np.all(actual_ids % 2 == 1)


def test_pyarrow_filters_make_reader(synthetic_dataset):
    with make_reader(synthetic_dataset.url, workers_count=5, num_epochs=1,
                     filters=[('partition_key', '=', 'p_5'), ]) as reader:
        uv = set()
        for data in reader:
            uv.add(data[0])

        assert uv == {'p_5'}


def test_transform_function_batched_auto_deleting_column(scalar_dataset):
    date_partition = pd.Timestamp(datetime.date(2019, 1, 3))
    with make_batch_reader(scalar_dataset.url
        , filters=[('datetime', '=', date_partition), ]) as reader:
        uv = set()
        for data in reader:
            for _datetime in data[0]:
                uv.add(_datetime)

        assert uv == {'2019-01-03'}
