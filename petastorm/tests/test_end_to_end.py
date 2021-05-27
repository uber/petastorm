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
import glob
import tempfile
import operator
import os
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
from petastorm.tests.test_common import create_test_dataset, TestSchema, create_test_scalar_dataset
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


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_simple_read_from_parquet_file(synthetic_dataset, reader_factory):
    """See if we can read data when a single parquet file is specified instead of a parquet directory"""
    assert synthetic_dataset.url.startswith('file://')
    path = synthetic_dataset.url[len('file://'):]
    one_parquet_file = glob.glob(f'{path}/**/*.parquet')[0]
    with reader_factory(f"file://{one_parquet_file}") as reader:
        all_data = list(reader)
        assert len(all_data) > 0


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


@pytest.mark.parametrize('reader_factory', [
    lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs)
])
def test_transform_function_returns_a_new_dict_with_predicate(synthetic_dataset, reader_factory):
    def transform(sample):
        return {'id': sample['id'], 'id2': -1}

    with reader_factory(synthetic_dataset.url, schema_fields=[TestSchema.id, TestSchema.id2],
                        predicate=in_lambda(['id2'], lambda id2: id2 == 1),
                        transform_spec=TransformSpec(func=transform)) as reader:
        rows = list(reader)
        actual_ids = np.asarray(list(row.id for row in rows))
        assert actual_ids.size > 0
        # In the test data id2 = id % 2, which means we expect only odd ids to remain after
        # we apply lambda id2: id2 == 1 predicate.
        assert np.all(actual_ids % 2 == 1)

        transformed_ids = np.asarray(list(row.id2 for row in rows))
        assert np.all(transformed_ids == -1)


@pytest.mark.parametrize('reader_factory', [
    lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs)
])
def test_transform_function_new_field(synthetic_dataset, reader_factory):
    """"""

    def double_matrix(sample):
        sample['double_matrix'] = sample['matrix'] * 2
        del sample['matrix']
        return sample

    with reader_factory(synthetic_dataset.url, schema_fields=[TestSchema.id, TestSchema.matrix],
                        transform_spec=TransformSpec(double_matrix,
                                                     [('double_matrix', np.float32, (32, 16, 3), False)],
                                                     ['matrix'])) as reader:
        actual = next(reader)
        original_sample = next(d for d in synthetic_dataset.data if d['id'] == actual.id)
        expected_matrix = original_sample['matrix'] * 2
        np.testing.assert_equal(expected_matrix, actual.double_matrix)


@mock.patch('petastorm.unischema._UNISCHEMA_FIELD_ORDER', 'alphabetical')
def test_transform_function_batched(scalar_dataset):
    def double_float64(sample):
        sample['float64'] *= 2
        return sample

    with make_batch_reader(scalar_dataset.url, transform_spec=TransformSpec(double_float64)) as reader:
        actual = next(reader)
        for actual_id, actual_float64 in zip(actual.id, actual.float64):
            original_sample = next(d for d in scalar_dataset.data if d['id'] == actual_id)
            expected_matrix = original_sample['float64'] * 2
            np.testing.assert_equal(expected_matrix, actual_float64)


def test_transform_function_batched_deleting_column(scalar_dataset):
    def double_float64(sample):
        del sample['float64']
        return sample

    with make_batch_reader(scalar_dataset.url,
                           transform_spec=TransformSpec(double_float64, removed_fields=['float64'])) as reader:
        actual = next(reader)
        assert 'float64' not in actual._fields


def test_transform_function_batched_auto_deleting_column(scalar_dataset):
    with make_batch_reader(scalar_dataset.url,
                           transform_spec=TransformSpec(removed_fields=['float64'])) as reader:
        actual = next(reader)
        assert 'float64' not in actual._fields


def test_transform_function_with_predicate_batched(scalar_dataset):
    def double_float64(sample):
        assert all(sample['id'] % 2 == 0)
        sample['float64'] *= 2
        return sample

    with make_batch_reader(scalar_dataset.url, transform_spec=TransformSpec(double_float64),
                           predicate=in_lambda(['id'], lambda id: id % 2 == 0)) as reader:
        actual = next(reader)
        for actual_id, actual_float64 in zip(actual.id, actual.float64):
            assert actual_id % 2 == 0
            original_sample = next(d for d in scalar_dataset.data if d['id'] == actual_id)
            expected_matrix = original_sample['float64'] * 2
            np.testing.assert_equal(expected_matrix, actual_float64)


@pytest.mark.parametrize('reader_factory', ALL_READER_FLAVOR_FACTORIES + SCALAR_ONLY_READER_FACTORIES)
@pytest.mark.forked
def test_simple_read_with_disk_cache(synthetic_dataset, reader_factory, tmpdir):
    """Try using the Reader with LocalDiskCache using different flavors of pools"""
    CACHE_SIZE = 10 * 2 ** 30  # 20GB
    ROW_SIZE_BYTES = 100  # not really important for this test
    with reader_factory(synthetic_dataset.url, num_epochs=2,
                        cache_type='local-disk', cache_location=tmpdir.strpath,
                        cache_size_limit=CACHE_SIZE, cache_row_size_estimate=ROW_SIZE_BYTES) as reader:
        ids = _readout_all_ids(reader)
        assert 200 == len(ids)  # We read 2 epochs
        assert set(ids) == set(range(100))


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES + SCALAR_ONLY_READER_FACTORIES)
def test_simple_read_with_added_slashes(synthetic_dataset, reader_factory):
    """Tests that using relative paths for the dataset metadata works as expected"""
    with reader_factory(synthetic_dataset.url + '///') as reader:
        next(reader)


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES + SCALAR_ONLY_READER_FACTORIES)
def test_simple_read_moved_dataset(synthetic_dataset, tmpdir, reader_factory):
    """Tests that a dataset may be opened after being moved to a new location"""
    a_moved_path = tmpdir.join('moved').strpath
    copytree(synthetic_dataset.path, a_moved_path)

    with reader_factory('file://{}'.format(a_moved_path)) as reader:
        next(reader)

    rmtree(a_moved_path)


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_reading_subset_of_columns(synthetic_dataset, reader_factory):
    """Just a bunch of read and compares of all values to the expected values"""
    with reader_factory(synthetic_dataset.url, schema_fields=[TestSchema.id2, TestSchema.id]) as reader:
        # Read a bunch of entries from the dataset and compare the data to reference
        for row in reader:
            actual = dict(row._asdict())
            expected = next(d for d in synthetic_dataset.data if d['id'] == actual['id'])
            np.testing.assert_equal(expected['id2'], actual['id2'])


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_reading_subset_of_columns_using_regex(synthetic_dataset, reader_factory):
    """Just a bunch of read and compares of all values to the expected values"""
    with reader_factory(synthetic_dataset.url, schema_fields=['id$', 'id_.*$', 'partition_key$']) as reader:
        # Read a bunch of entries from the dataset and compare the data to reference
        for row in reader:
            actual = dict(row._asdict())
            assert set(actual.keys()) == {'id_float', 'id_odd', 'id', 'partition_key'}
            expected = next(d for d in synthetic_dataset.data if d['id'] == actual['id'])
            np.testing.assert_equal(expected['id_float'], actual['id_float'])


@pytest.mark.parametrize('reader_factory', [
    lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs),
    lambda url, **kwargs: make_batch_reader(url, reader_pool_type='dummy', **kwargs)])
def test_shuffle(synthetic_dataset, reader_factory):
    rows_count = len(synthetic_dataset.data)

    # Read ids twice without shuffle: assert we have the same array and all expected ids are in the array
    with reader_factory(synthetic_dataset.url, shuffle_row_groups=False) as reader_1:
        first_readout = _readout_all_ids(reader_1)
    with reader_factory(synthetic_dataset.url, shuffle_row_groups=False) as reader_2:
        second_readout = _readout_all_ids(reader_2)

    np.testing.assert_array_equal(range(rows_count), sorted(first_readout))
    np.testing.assert_array_equal(first_readout, second_readout)

    # Now read with shuffling
    with reader_factory(synthetic_dataset.url, shuffle_row_groups=True) as shuffled_reader:
        shuffled_readout = _readout_all_ids(shuffled_reader)
    assert np.any(np.not_equal(first_readout, shuffled_readout))


@pytest.mark.parametrize('reader_factory', [
    lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs),
    lambda url, **kwargs: make_batch_reader(url, reader_pool_type='dummy', **kwargs)])
def test_shuffle_drop_ratio(synthetic_dataset, reader_factory):
    # Read ids twice without shuffle: assert we have the same array and all expected ids are in the array
    with reader_factory(synthetic_dataset.url, shuffle_row_groups=False, shuffle_row_drop_partitions=1) as reader:
        first_readout = _readout_all_ids(reader)
    np.testing.assert_array_equal([r['id'] for r in synthetic_dataset.data], sorted(first_readout))

    # Test that the ids are increasingly not consecutive numbers as we increase the shuffle dropout
    prev_jumps_not_1 = 0
    for shuffle_dropout in [2, 5, 8]:
        with reader_factory(synthetic_dataset.url, shuffle_row_groups=True,
                            shuffle_row_drop_partitions=shuffle_dropout) as reader:
            readout = _readout_all_ids(reader)

        assert len(first_readout) == len(readout)
        jumps_not_1 = np.sum(np.diff(readout) != 1)
        assert jumps_not_1 > prev_jumps_not_1
        prev_jumps_not_1 = jumps_not_1


@pytest.mark.parametrize('reader_factory', ALL_READER_FLAVOR_FACTORIES)
def test_predicate_on_partition(synthetic_dataset, reader_factory):
    for expected_partition_keys in [{'p_0', 'p_2'}, {'p_0'}, {'p_1', 'p_2'}]:
        with reader_factory(synthetic_dataset.url,
                            predicate=PartitionKeyInSetPredicate(expected_partition_keys)) as reader:
            partition_keys = set(row.partition_key for row in reader)
            assert partition_keys == expected_partition_keys


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_predicate_on_partition_filters_out_everything(synthetic_dataset, reader_factory):
    with pytest.warns(UserWarning, match='No matching data is available for loading'):
        # This predicate should filter out all rowgroups. We should raise an error in this case.
        make_reader(synthetic_dataset.url, reader_pool_type='dummy',
                    predicate=PartitionKeyInSetPredicate({'non existing value'}))


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_too_many_shards(synthetic_dataset, reader_factory):
    with pytest.raises(NoDataAvailableError, match='Number of row-groups in the dataset'):
        # If number of shards is greater than number of rowgroups, users might be surprised if a reader
        # does not produce any error, hence we raise an explicit exception
        make_reader(synthetic_dataset.url, reader_pool_type='dummy', cur_shard=0, shard_count=10000000)


@pytest.mark.parametrize('reader_factory', SCALAR_ONLY_READER_FACTORIES)
def test_predicate_on_partition_batched(synthetic_dataset, reader_factory):
    for expected_partition_keys in [{'p_0', 'p_2'}, {'p_0'}, {'p_1', 'p_2'}]:
        # TODO(yevgeni): scalar only reader takes 'vectorized' predicate that processes entire columns. Not
        # yet implemented for the case of a prediction on partition, hence we use a non-vectorized
        # PartitionKeyInSetPredicate here
        with reader_factory(synthetic_dataset.url,
                            predicate=PartitionKeyInSetPredicate(expected_partition_keys)) as reader:
            partition_keys = set()
            for row in reader:
                partition_keys |= set(row.partition_key)
            assert partition_keys == expected_partition_keys


@pytest.mark.parametrize('reader_factory', ALL_READER_FLAVOR_FACTORIES)
def test_predicate_on_multiple_fields(synthetic_dataset, reader_factory):
    expected_values = {'id': 11, 'id2': 1}
    with reader_factory(synthetic_dataset.url, shuffle_row_groups=False,
                        predicate=EqualPredicate(expected_values)) as reader:
        actual = next(reader)
        assert actual.id == expected_values['id']
        assert actual.id2 == expected_values['id2']


@pytest.mark.parametrize('reader_factory', SCALAR_ONLY_READER_FACTORIES)
def test_predicate_on_multiple_fields_batched(synthetic_dataset, reader_factory):
    expected_values = {'id': 11, 'id2': 1}
    with reader_factory(synthetic_dataset.url, shuffle_row_groups=False,
                        predicate=VectorizedEqualPredicate(expected_values)) as reader:
        actual = next(reader)
        assert actual.id.shape == (1,)
        assert actual.id[0] == expected_values['id']
        assert actual.id2[0] == expected_values['id2']


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES + SCALAR_ONLY_READER_FACTORIES)
def test_predicate_with_invalid_fields(synthetic_dataset, reader_factory):
    """Try passing an invalid field name from a predicate to the reader. An error should be raised."""
    TEST_CASES = [
        {'invalid_field_name': 1},
        dict(),
        {'invalid_field_name': 1, 'id': 11},
        {'invalid_field_name': 1, 'invalid_field_name_2': 11}]

    for predicate_spec in TEST_CASES:
        with reader_factory(synthetic_dataset.url, shuffle_row_groups=False,
                            predicate=EqualPredicate(predicate_spec)) as reader:
            with pytest.raises(ValueError):
                next(reader)


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES + SCALAR_ONLY_READER_FACTORIES)
@pytest.mark.parametrize('shard_seed', [None, 0])
def test_partition_multi_node(synthetic_dataset, reader_factory, shard_seed):
    """Tests that the reader only returns half of the expected data consistently"""
    with reader_factory(synthetic_dataset.url, cur_shard=0, shard_count=5, shard_seed=shard_seed) as reader:
        with reader_factory(synthetic_dataset.url, cur_shard=0, shard_count=5, shard_seed=shard_seed) as reader_2:
            results_1 = set(_readout_all_ids(reader))
            results_2 = set(_readout_all_ids(reader_2))

            assert results_1, 'Non empty shard expected'

            np.testing.assert_equal(results_1, results_2)

            assert len(results_1) < len(synthetic_dataset.data)

            # Test that separate partitions also have no overlap by checking ids)
            for partition in range(1, 5):
                with reader_factory(synthetic_dataset.url, cur_shard=partition,
                                    shard_count=5, shard_seed=shard_seed) as reader_other:
                    ids_in_other_partition = set(_readout_all_ids(reader_other))

                    assert not ids_in_other_partition.intersection(results_1)


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES + SCALAR_ONLY_READER_FACTORIES)
def test_partition_value_error(synthetic_dataset, reader_factory):
    """Tests that the reader raises value errors when appropriate"""

    # shard_count has to be greater than 0
    with pytest.raises(ValueError):
        reader_factory(synthetic_dataset.url, shard_count=0)

    # missing cur_shard value
    with pytest.raises(ValueError):
        reader_factory(synthetic_dataset.url, shard_count=5)

    # cur_shard is a string
    with pytest.raises(ValueError):
        reader_factory(synthetic_dataset.url, cur_shard='0',
                       shard_count=5)

    # shard_count is a string
    with pytest.raises(ValueError):
        reader_factory(synthetic_dataset.url, cur_shard=0,
                       shard_count='5')


@pytest.mark.parametrize('reader_factory', [
    lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs),
    lambda url, **kwargs: make_batch_reader(url, reader_pool_type='dummy', **kwargs),
])
def test_stable_pieces_order(synthetic_dataset, reader_factory):
    """Tests that the reader raises value errors when appropriate"""

    RERUN_THE_TEST_COUNT = 4
    baseline_run = None
    for _ in range(RERUN_THE_TEST_COUNT):
        # TODO(yevgeni): factor out. Reading all ids appears multiple times in this test.
        with reader_factory(synthetic_dataset.url, shuffle_row_groups=False) as reader:
            this_run = _readout_all_ids(reader)

        if baseline_run:
            assert this_run == baseline_run

        baseline_run = this_run


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_invalid_schema_field(synthetic_dataset, reader_factory):
    # Let's assume we are selecting columns using a schema which is different from the one
    # stored in the dataset. Would expect to get a reasonable error message
    BogusSchema = Unischema('BogusSchema', [
        UnischemaField('partition_key', np.string_, (), ScalarCodec(StringType()), False),
        UnischemaField('id', np.int64, (), ScalarCodec(LongType()), False),
        UnischemaField('bogus_key', np.int32, (), ScalarCodec(ShortType()), False)])

    expected_values = {'bogus_key': 11, 'id': 1}
    with pytest.raises(ValueError, match='bogus_key'):
        reader_factory(synthetic_dataset.url, schema_fields=BogusSchema.fields.values(),
                       shuffle_row_groups=False,
                       predicate=EqualPredicate(expected_values))


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_use_persisted_codec_and_not_provided_by_user(synthetic_dataset, reader_factory):
    """In order to start using new codec for some field while maintain the ability to read old datasets that were
    written using an old codec, we need to make sure we are using stored UnischemaField.codec object (that contains
    an old codec/shape)."""
    new_unischema_instance = UnischemaField('matrix_uint16', np.uint16, (2, 3, 4), CompressedImageCodec('png'), False)

    with reader_factory(synthetic_dataset.url, schema_fields=[new_unischema_instance]) as reader:
        row = next(reader)
    assert row.matrix_uint16.shape == (32, 16, 3)


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_single_column_predicate(synthetic_dataset, reader_factory):
    """Test quering a single column with a predicate on the same column """
    with reader_factory(synthetic_dataset.url, schema_fields=[TestSchema.id], predicate=EqualPredicate({'id': 1})) \
            as reader:
        all_rows = list(reader)
        assert 1 == len(all_rows)
        assert 1 == all_rows[0].id


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_two_column_predicate(synthetic_dataset, reader_factory):
    """Test quering a single column with a predicate on the same column """
    with reader_factory(synthetic_dataset.url, schema_fields=[TestSchema.id2, TestSchema.partition_key],
                        predicate=EqualPredicate({'id2': 1, 'partition_key': 'p_2'})) as reader:
        all_rows = list(reader)
        all_id2 = np.array(list(map(operator.attrgetter('id2'), all_rows)))
        all_partition_key = np.array(list(map(operator.attrgetter('partition_key'), all_rows)))
        assert (all_id2 == 1).all()
        assert (all_partition_key == 'p_2').all()


@pytest.mark.parametrize('reader_factory',
                         [lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs),
                          lambda url, **kwargs: make_reader(url, reader_pool_type='thread', **kwargs),
                          lambda url, **kwargs: make_reader(url, reader_pool_type='process', **kwargs)])
def test_multiple_epochs(synthetic_dataset, reader_factory):
    """Tests that multiple epochs works as expected"""
    num_epochs = 5
    with reader_factory(synthetic_dataset.url, num_epochs=num_epochs) as reader:
        # Read all expected entries from the dataset and compare the data to reference
        single_epoch_id_set = [d['id'] for d in synthetic_dataset.data]
        actual_ids_in_all_epochs = _readout_all_ids(reader)
        np.testing.assert_equal(sorted(actual_ids_in_all_epochs), sorted(num_epochs * single_epoch_id_set))

        # Reset reader should reset ventilator. Should produce another `num_epochs` results
        reader.reset()
        actual_ids_in_all_epochs = _readout_all_ids(reader)
        np.testing.assert_equal(sorted(actual_ids_in_all_epochs), sorted(num_epochs * single_epoch_id_set))


@pytest.mark.parametrize('reader_factory',
                         [lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs),
                          lambda url, **kwargs: make_reader(url, reader_pool_type='thread', **kwargs),
                          lambda url, **kwargs: make_reader(url, reader_pool_type='process', **kwargs)])
def test_fail_if_resetting_in_the_middle_of_epoch(synthetic_dataset, reader_factory):
    """Tests that multiple epochs works as expected"""
    num_epochs = 5
    with reader_factory(synthetic_dataset.url, num_epochs=num_epochs) as reader:
        # Read all expected entries from the dataset and compare the data to reference
        actual_ids = _readout_all_ids(reader, limit=20)
        assert len(actual_ids) == 20

        with pytest.raises(NotImplementedError):
            reader.reset()


# TODO(yevgeni) this test is broken for reader_v2
@pytest.mark.parametrize('reader_factory', [MINIMAL_READER_FLAVOR_FACTORIES[0]] + SCALAR_ONLY_READER_FACTORIES)
def test_unlimited_epochs(synthetic_dataset, reader_factory):
    """Tests that unlimited epochs works as expected"""
    with reader_factory(synthetic_dataset.url, num_epochs=None) as reader:
        read_limit = len(synthetic_dataset.data) * 3 + 2
        actual_ids = _readout_all_ids(reader, read_limit)
        expected_ids = [d['id'] for d in synthetic_dataset.data]
        assert len(actual_ids) > len(expected_ids)
        assert set(actual_ids) == set(expected_ids)


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES + SCALAR_ONLY_READER_FACTORIES)
def test_num_epochs_value_error(synthetic_dataset, reader_factory):
    """Tests that the reader raises value errors when appropriate"""

    # Testing only Reader v1, as the v2 uses an epoch generator. The error would raise only when the generator is
    # evaluated. Parameter validation for Reader v2 is covered by test_epoch_generator.py

    with pytest.raises(ValueError):
        reader_factory(synthetic_dataset.url, num_epochs=-10)

    with pytest.raises(ValueError):
        reader_factory(synthetic_dataset.url, num_epochs='abc')


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_rowgroup_selector_integer_field(synthetic_dataset, reader_factory):
    """ Select row groups to read based on dataset index for integer field"""
    with reader_factory(synthetic_dataset.url, rowgroup_selector=SingleIndexSelector(TestSchema.id.name, [2, 18])) \
            as reader:
        status = [False, False]
        count = 0
        for row in reader:
            if row.id == 2:
                status[0] = True
            if row.id == 18:
                status[1] = True
            count += 1
        # both id values in reader result
        assert all(status)
        # read only 2 row groups, 100 rows per row group
        assert 20 == count


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_rowgroup_selector_string_field(synthetic_dataset, reader_factory):
    """ Select row groups to read based on dataset index for string field"""
    with reader_factory(synthetic_dataset.url,
                        rowgroup_selector=SingleIndexSelector(TestSchema.sensor_name.name, ['test_sensor'])) as reader:
        count = sum(1 for _ in reader)

        # Since we use artificial dataset all sensors have the same name,
        # so all row groups should be selected and all 1000 generated rows should be returned
        assert 100 == count


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_rowgroup_selector_multiple_fields_intersection(synthetic_dataset, reader_factory):
    intersect_index_selector = IntersectIndexSelector(
        [SingleIndexSelector(TestSchema.sensor_name.name, ['test_sensor']),
         SingleIndexSelector(TestSchema.id.name, [2, 18])]
    )
    with reader_factory(synthetic_dataset.url,
                        rowgroup_selector=intersect_index_selector) as reader:
        count = 0
        status = [False, False, False]
        for row in reader:
            if row.id == 2:
                status[0] = True
            if row.id == 18:
                status[1] = True
            if row.sensor_name == 'test_sensor':
                status[2] = True
            count += 1
        assert all(status)
        assert 20 == count


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_rowgroup_selector_multiple_fields_union(synthetic_dataset, reader_factory):
    union_index_selector = UnionIndexSelector(
        [SingleIndexSelector(TestSchema.sensor_name.name, ['test_sensor']),
         SingleIndexSelector(TestSchema.id.name, [2, 18])]
    )
    with reader_factory(synthetic_dataset.url,
                        rowgroup_selector=union_index_selector) as reader:
        count = 0
        status = [False, False, False]
        for row in reader:
            if row.id == 2:
                status[0] = True
            if row.id == 18:
                status[1] = True
            if row.sensor_name == 'test_sensor':
                status[2] = True
            count += 1
        assert all(status)
        assert 100 == count


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_rowgroup_selector_nullable_array_field(synthetic_dataset, reader_factory):
    """ Select row groups to read based on dataset index for array field"""
    with reader_factory(synthetic_dataset.url,
                        rowgroup_selector=SingleIndexSelector(TestSchema.string_array_nullable.name,
                                                              ['100'])) as reader:
        count = sum(1 for _ in reader)
        # This field contain id string, generated like this
        #   None if id % 5 == 0 else np.asarray([], dtype=np.string_) if id % 4 == 0 else
        #   np.asarray([str(i+id) for i in xrange(2)], dtype=np.string_)
        # hence '100' could be present in row id 99 as 99+1 and row id 100 as 100+0
        # but row 100 will be skipped by ' None if id % 5 == 0' condition, so only one row group should be selected
        assert 10 == count


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_rowgroup_selector_partition_key(synthetic_dataset, reader_factory):
    """ Select row groups to read based on dataset index for array field"""
    with reader_factory(synthetic_dataset.url,
                        rowgroup_selector=SingleIndexSelector(TestSchema.partition_key.name,
                                                              ['p_1'])) as reader:
        count = sum(1 for _ in reader)
        assert 10 == count


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_rowgroup_selector_wrong_index_name(synthetic_dataset, reader_factory):
    """ Attempt to select row groups to based on wrong dataset index,
        Reader should raise exception
    """
    with pytest.raises(ValueError):
        reader_factory(synthetic_dataset.url, rowgroup_selector=SingleIndexSelector('WrongIndexName', ['some_value']))


def test_materialize_dataset_hadoop_config(tmpdir_factory):
    """Test that using materialize_dataset does not alter the hadoop_config"""

    path = tmpdir_factory.mktemp('data').strpath
    tmp_url = "file://" + path
    # This test does not properly check if parquet.enable.summary-metadata is restored properly with pyspark < 2.4
    spark = SparkSession.builder.getOrCreate()
    hadoop_config = spark.sparkContext._jsc.hadoopConfiguration()

    parquet_metadata_level = "COMMON_ONLY"
    parquet_row_group_check = 100

    # Set the parquet summary files and row group size check min
    hadoop_config.set('parquet.summary.metadata.level', parquet_metadata_level)
    hadoop_config.setInt('parquet.row-group.size.row.check.min', parquet_row_group_check)
    assert hadoop_config.get('parquet.summary.metadata.level') == str(parquet_metadata_level)
    assert hadoop_config.get('parquet.row-group.size.row.check.min') == str(parquet_row_group_check)

    create_test_dataset(tmp_url, range(10), spark=spark)

    assert not os.path.exists(os.path.join(path, "_metadata"))

    # Check that they are back to the original values after writing the dataset
    hadoop_config = spark.sparkContext._jsc.hadoopConfiguration()
    assert hadoop_config.get('parquet.summary.metadata.level') == str(parquet_metadata_level)
    assert hadoop_config.get('parquet.row-group.size.row.check.min') == str(parquet_row_group_check)
    # Other options should return to being unset
    assert hadoop_config.get('parquet.block.size') is None
    assert hadoop_config.get('parquet.block.size.row.check.min') is None
    spark.stop()


def test_materialize_with_summary_metadata(tmpdir_factory):
    """Verify _summary_metadata appears, when requested"""
    path = tmpdir_factory.mktemp('data').strpath
    tmp_url = "file://" + path

    spark = SparkSession.builder.getOrCreate()
    create_test_dataset(tmp_url, range(10), spark=spark, use_summary_metadata=True)

    assert os.path.exists(os.path.join(path, "_metadata"))
    spark.stop()


def test_pass_in_pyarrow_filesystem_to_materialize_dataset(synthetic_dataset, tmpdir):
    a_moved_path = tmpdir.join('moved').strpath
    copytree(synthetic_dataset.path, a_moved_path)

    local_fs = pyarrow.LocalFileSystem
    os.remove(a_moved_path + '/_common_metadata')

    spark = SparkSession.builder.getOrCreate()

    with materialize_dataset(spark, a_moved_path, TestSchema, filesystem_factory=local_fs):
        pass

    with make_reader('file://{}'.format(a_moved_path), reader_pool_type='dummy') as reader:
        _check_simple_reader(reader, synthetic_dataset.data)

    spark.stop()
    rmtree(a_moved_path)


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES + SCALAR_ONLY_READER_FACTORIES)
def test_dataset_path_is_a_unicode(synthetic_dataset, reader_factory):
    """Just a bunch of read and compares of all values to the expected values using the different reader pools"""
    # Making sure unicode_in_p23 is a unicode both in python 2 and 3
    unicode_in_p23 = synthetic_dataset.url.encode().decode('utf-8')
    with reader_factory(unicode_in_p23) as reader:
        next(reader)


def test_multithreaded_reads(synthetic_dataset):
    with make_reader(synthetic_dataset.url, workers_count=5, num_epochs=1) as reader:
        with ThreadPoolExecutor(max_workers=10) as executor:
            def read_one_row():
                return next(reader)

            futures = [executor.submit(read_one_row) for _ in range(100)]
            results = [f.result() for f in futures]
            assert len(results) == len(synthetic_dataset.data)
            assert set(r.id for r in results) == set(d['id'] for d in synthetic_dataset.data)


def test_should_fail_if_reading_out_of_context_manager(synthetic_dataset):
    with make_reader(synthetic_dataset.url, workers_count=1) as reader:
        next(reader)

    with pytest.raises(RuntimeError, match='Trying to read a sample.*'):
        next(reader)


def test_should_fail_if_reading_after_stop(synthetic_dataset):
    reader = make_reader(synthetic_dataset.url, workers_count=1)
    next(reader)
    reader.stop()

    with pytest.raises(RuntimeError, match='Trying to read a sample.*'):
        next(reader)


def _get_local_fs_url_list(dir_url):
    url_list = []
    dir_path = urlparse(dir_url).path
    for file_name in os.listdir(dir_path):
        url_list.append('file://{dir_path}/{file_name}'.format(dir_path=dir_path, file_name=file_name))
    return url_list


def test_make_batch_reader_with_url_list(scalar_dataset):
    url_list = _get_local_fs_url_list(scalar_dataset.url)
    url_list = list(filter(lambda x: x.endswith('.parquet'), url_list))

    with make_batch_reader(url_list, workers_count=1) as reader:
        row_count = 0
        for batch in reader:
            row_count += len(batch.id)

        assert row_count == 100


def test_pyarrow_filters_make_reader(synthetic_dataset):
    with make_reader(synthetic_dataset.url, workers_count=5, num_epochs=1,
                     filters=[('partition_key', '=', 'p_5'), ]) as reader:
        uv = set()
        for data in reader:
            uv.add(data.partition_key)

        assert uv == {'p_5'}


def test_pyarrow_filters_make_batch_reader():
    path = tempfile.mkdtemp()
    url = 'file://' + path
    create_test_scalar_dataset(url, 3000, partition_by=['id_div_700'])
    with make_batch_reader(url, filters=[('id_div_700', '=', 2), ]) as reader:
        uv = set()
        for data in reader:
            for _id_div_700 in data.id_div_700:
                uv.add(_id_div_700)

        assert uv == {2}
