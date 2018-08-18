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
import random
from shutil import rmtree, copytree

import numpy as np
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType, ShortType, StringType

from petastorm.codecs import ScalarCodec
from petastorm.local_disk_cache import LocalDiskCache
from petastorm.reader import Reader, ShuffleOptions
from petastorm.selectors import SingleIndexSelector
from petastorm.tests.test_common import create_test_dataset, TestSchema
from petastorm.tests.test_end_to_end_predicates_impl import \
    PartitionKeyInSetPredicate, EqualPredicate
from petastorm.unischema import UnischemaField, Unischema
from petastorm.workers_pool.dummy_pool import DummyPool
from petastorm.workers_pool.process_pool import ProcessPool
from petastorm.workers_pool.thread_pool import ThreadPool

# Number of rows in a fake dataset

# pylint: disable=unnecessary-lambda
ALL_READER_FLAVOR_FACTORIES = [lambda url, **kwargs: Reader(url, reader_pool=DummyPool(), **kwargs),
                               lambda url, **kwargs: Reader(url, reader_pool=ThreadPool(10), **kwargs),
                               lambda url, **kwargs: Reader(url, reader_pool=ProcessPool(10), **kwargs)]


def _check_simple_reader(reader, expected_data):
    # Read a bunch of entries from the dataset and compare the data to reference
    for row in reader:
        actual = dict(row._asdict())
        expected = next(d for d in expected_data if d['id'] == actual['id'])
        np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize('reader_factory', ALL_READER_FLAVOR_FACTORIES)
def test_simple_read(synthetic_dataset, reader_factory):
    """Just a bunch of read and compares of all values to the expected values using the different reader pools"""
    with reader_factory(synthetic_dataset.url) as reader:
        _check_simple_reader(reader, synthetic_dataset.data)


@pytest.mark.parametrize('reader_factory', ALL_READER_FLAVOR_FACTORIES)
def test_simple_read_with_disk_cache(synthetic_dataset, reader_factory, tmpdir):
    """Try using the Reader with LocalDiskCache using different flavors of pools"""
    CACHE_SIZE = 10 * 2 ** 30  # 20GB
    ROW_SIZE_BYTES = 100  # not really important for this test
    with reader_factory(synthetic_dataset.url, num_epochs=2,
                        cache=LocalDiskCache(tmpdir.strpath, CACHE_SIZE, ROW_SIZE_BYTES)) as reader:
        _check_simple_reader(reader, synthetic_dataset.data)


def test_simple_read_with_added_slashes(synthetic_dataset):
    """Tests that using relative paths for the dataset metadata works as expected"""
    with Reader(synthetic_dataset.url + '///', reader_pool=DummyPool()) as reader:
        _check_simple_reader(reader, synthetic_dataset.data)


def test_simple_read_moved_dataset(synthetic_dataset, tmpdir):
    """Tests that a dataset may be opened after being moved to a new location"""
    a_moved_path = tmpdir.join('moved').strpath
    copytree(synthetic_dataset.path, a_moved_path)

    with Reader('file://{}'.format(a_moved_path), reader_pool=DummyPool()) as reader:
        _check_simple_reader(reader, synthetic_dataset.data)


def test_reading_subset_of_columns(synthetic_dataset):
    """Just a bunch of read and compares of all values to the expected values"""
    with Reader(synthetic_dataset.url, schema_fields=[TestSchema.id2, TestSchema.id],
                reader_pool=DummyPool()) as reader:
        # Read a bunch of entries from the dataset and compare the data to reference
        for row in reader:
            actual = dict(row._asdict())
            expected = next(d for d in synthetic_dataset.data if d['id'] == actual['id'])
            np.testing.assert_equal(expected['id2'], actual['id2'])


def test_shuffle(synthetic_dataset):
    rows_count = len(synthetic_dataset.data)

    def readout_all_ids(shuffle):
        with Reader(synthetic_dataset.url,
                    shuffle_options=ShuffleOptions(shuffle),
                    reader_pool=ThreadPool(1)) as reader:
            ids = [row.id for row in reader]
        return ids

    # Read ids twice without shuffle: assert we have the same array and all expected ids are in the array
    first_readout = readout_all_ids(False)
    second_readout = readout_all_ids(False)
    np.testing.assert_array_equal(range(rows_count), sorted(first_readout))
    np.testing.assert_array_equal(first_readout, second_readout)

    # Now read with shuffling
    shuffled_readout = readout_all_ids(True)
    assert np.any(np.not_equal(first_readout, shuffled_readout))


def test_shuffle_drop_ratio(synthetic_dataset):
    def readout_all_ids(shuffle, drop_ratio):
        with Reader(dataset_url=synthetic_dataset.url,
                    reader_pool=DummyPool(),
                    shuffle_options=ShuffleOptions(shuffle, drop_ratio)) as reader:
            ids = [row.id for row in reader]
        return ids

    # Read ids twice without shuffle: assert we have the same arra  y and all expected ids are in the array
    first_readout = readout_all_ids(False, 1)
    np.testing.assert_array_equal([r['id'] for r in synthetic_dataset.data], sorted(first_readout))

    # Test that the ids are increasingly not consecutive numbers as we increase the shuffle dropout
    prev_jumps_not_1 = 0
    for shuffle_dropout in [2, 5, 8, 111]:
        readout = readout_all_ids(True, shuffle_dropout)
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


@pytest.mark.parametrize('reader_factory', ALL_READER_FLAVOR_FACTORIES)
def test_predicate_on_multiple_fields(synthetic_dataset, reader_factory):
    expected_values = {'id': 11, 'id2': 1}
    with reader_factory(synthetic_dataset.url, shuffle_options=ShuffleOptions(False),
                        predicate=EqualPredicate(expected_values)) as reader:
        actual = next(reader)
        assert actual.id == expected_values['id']
        assert actual.id2 == expected_values['id2']


def test_predicate_with_invalid_fields(synthetic_dataset):
    """Try passing an invalid field name from a predicate to the reader. An error should be raised."""
    TEST_CASES = [
        {'invalid_field_name': 1},
        dict(),
        {'invalid_field_name': 1, 'id': 11},
        {'invalid_field_name': 1, 'invalid_field_name_2': 11}]

    for predicate_spec in TEST_CASES:
        with Reader(synthetic_dataset.url, shuffle_options=ShuffleOptions(False),
                    predicate=EqualPredicate(predicate_spec),
                    reader_pool=ThreadPool(1)) as reader:
            with pytest.raises(ValueError):
                next(reader)


def test_partition_multi_node(synthetic_dataset):
    """Tests that the reader only returns half of the expected data consistently"""
    reader = Reader(synthetic_dataset.url, reader_pool=DummyPool(), training_partition=0,
                    num_training_partitions=5)
    reader_2 = Reader(synthetic_dataset.url, reader_pool=DummyPool(), training_partition=0,
                      num_training_partitions=5)

    results_1 = []
    expected = []
    for row in reader:
        actual = dict(row._asdict())
        results_1.append(actual)
        expected.append(next(d for d in synthetic_dataset.data if d['id'] == actual['id']))

    results_2 = [dict(row._asdict()) for row in reader_2]

    # Since order is non deterministic, we need to sort results by id
    results_1.sort(key=lambda x: x['id'])
    results_2.sort(key=lambda x: x['id'])
    expected.sort(key=lambda x: x['id'])

    np.testing.assert_equal(expected, results_1)
    np.testing.assert_equal(results_1, results_2)

    assert len(results_1) < len(synthetic_dataset.data)

    # Test that separate partitions also have no overlap by checking ids
    id_set = set([item['id'] for item in results_1])
    for partition in range(1, 5):
        with Reader(synthetic_dataset.url, reader_pool=DummyPool(), training_partition=partition,
                    num_training_partitions=5) as reader_other:

            for row in reader_other:
                assert dict(row._asdict())['id'] not in id_set

    reader.stop()
    reader.join()
    reader_2.stop()
    reader_2.join()


def test_partition_value_error(synthetic_dataset):
    """Tests that the reader raises value errors when appropriate"""

    with pytest.raises(ValueError):
        Reader(synthetic_dataset.url, reader_pool=DummyPool(), training_partition=0)

    with pytest.raises(ValueError):
        Reader(synthetic_dataset.url, reader_pool=DummyPool(), num_training_partitions=5)

    with pytest.raises(ValueError):
        Reader(synthetic_dataset.url, reader_pool=DummyPool(), training_partition='0',
               num_training_partitions=5)

    with pytest.raises(ValueError):
        Reader(synthetic_dataset.url, reader_pool=DummyPool(), training_partition=0,
               num_training_partitions='5')


def test_stable_pieces_order(synthetic_dataset):
    """Tests that the reader raises value errors when appropriate"""

    RERUN_THE_TEST_COUNT = 20
    baseline_run = None
    for _ in range(RERUN_THE_TEST_COUNT):
        with Reader(synthetic_dataset.url, schema_fields=[TestSchema.id], shuffle_options=ShuffleOptions(False),
                    reader_pool=DummyPool()) as reader:
            this_run = [row.id for row in reader]
        if baseline_run:
            assert this_run == baseline_run

        baseline_run = this_run


def test_invalid_schema_field(synthetic_dataset):
    # Let's assume we are selecting columns using a schema which is different from the one
    # stored in the dataset. Would expect to get a reasonable error message
    BogusSchema = Unischema('BogusSchema', [
        UnischemaField('partition_key', np.string_, (), ScalarCodec(StringType()), False),
        UnischemaField('id', np.int64, (), ScalarCodec(LongType()), False),
        UnischemaField('bogus_key', np.int32, (), ScalarCodec(ShortType()), False)])

    expected_values = {'bogus_key': 11, 'id': 1}
    with pytest.raises(ValueError) as e:
        Reader(synthetic_dataset.url, schema_fields=BogusSchema.fields.values(), shuffle_options=ShuffleOptions(False),
               predicate=EqualPredicate(expected_values), reader_pool=ThreadPool(1))

    assert 'bogus_key' in str(e)


def test_single_column_predicate(synthetic_dataset):
    """Test quering a single column with a predicate on the same column """
    with Reader(synthetic_dataset.url, schema_fields=[TestSchema.id], predicate=EqualPredicate({'id': 1}),
                reader_pool=ThreadPool(1)) as reader:
        # Read a bunch of entries from the dataset and compare the data to reference
        for row in reader:
            actual = dict(row._asdict())
            expected = next(d for d in synthetic_dataset.data if d['id'] == actual['id'])
            np.testing.assert_equal(expected['id'], actual['id'])


def test_multiple_epochs(synthetic_dataset):
    """Tests that multiple epochs works as expected"""
    num_epochs = 5
    with Reader(synthetic_dataset.url, reader_pool=DummyPool(), num_epochs=num_epochs) as reader:
        # Read all expected entries from the dataset and compare the data to reference
        id_set = set([d['id'] for d in synthetic_dataset.data])

        for _ in range(num_epochs):
            current_epoch_set = set()
            for _ in range(len(id_set)):
                actual = dict(next(reader)._asdict())
                expected = next(d for d in synthetic_dataset.data if d['id'] == actual['id'])
                np.testing.assert_equal(expected, actual)
                current_epoch_set.add(actual['id'])
            np.testing.assert_equal(id_set, current_epoch_set)


def test_unlimited_epochs(synthetic_dataset):
    """Tests that unlimited epochs works as expected"""
    with Reader(synthetic_dataset.url, reader_pool=DummyPool(), num_epochs=None) as reader:
        # Read many expected entries from the dataset and compare the data to reference
        for _ in range(len(synthetic_dataset.data) * random.randint(10, 30) + random.randint(25, 50)):
            actual = dict(next(reader)._asdict())
            expected = next(d for d in synthetic_dataset.data if d['id'] == actual['id'])
            np.testing.assert_equal(expected, actual)


def test_num_epochs_value_error(synthetic_dataset):
    """Tests that the reader raises value errors when appropriate"""

    with pytest.raises(ValueError):
        Reader(synthetic_dataset.url, reader_pool=DummyPool(), num_epochs=0)

    with pytest.raises(ValueError):
        Reader(synthetic_dataset.url, reader_pool=DummyPool(), num_epochs=-10)

    with pytest.raises(ValueError):
        Reader(synthetic_dataset.url, reader_pool=DummyPool(), num_epochs='abc')


def test_rowgroup_selector_integer_field(synthetic_dataset):
    """ Select row groups to read based on dataset index for integer field"""
    with Reader(synthetic_dataset.url, rowgroup_selector=SingleIndexSelector(TestSchema.id.name, [2, 18]),
                reader_pool=DummyPool()) as reader:
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
        # read only 2 row groups, 10 rows per row group
        assert 20 == count


def test_rowgroup_selector_string_field(synthetic_dataset):
    """ Select row groups to read based on dataset index for string field"""
    with Reader(synthetic_dataset.url,
                rowgroup_selector=SingleIndexSelector(TestSchema.sensor_name.name, ['test_sensor']),
                reader_pool=DummyPool()) as reader:
        count = 0
        for _ in reader:
            count += 1
        # Since we use artificial dataset all sensors have the same name,
        # so all row groups should be selected and all 100 generated rows should be returned
        assert 100 == count


def test_rowgroup_selector_nullable_array_field(synthetic_dataset):
    """ Select row groups to read based on dataset index for array field"""
    with Reader(synthetic_dataset.url,
                rowgroup_selector=SingleIndexSelector(TestSchema.string_array_nullable.name, ['100']),
                reader_pool=DummyPool()) as reader:
        count = sum(1 for _ in reader)
        # This field contain id string, generated like this
        #   None if id % 5 == 0 else np.asarray([], dtype=np.string_) if id % 4 == 0 else
        #   np.asarray([str(i+id) for i in xrange(2)], dtype=np.string_)
        # hence '100' could be present in row id 99 as 99+1 and row id 100 as 100+0
        # but row 100 will be skipped by ' None if id % 5 == 0' condition, so only one row group should be selected
        assert 10 == count


def test_rowgroup_selector_wrong_index_name(synthetic_dataset):
    """ Attempt to select row groups to based on wrong dataset index,
        Reader should raise exception
    """
    with pytest.raises(ValueError):
        Reader(synthetic_dataset.url,
               rowgroup_selector=SingleIndexSelector('WrongIndexName', ['some_value']), reader_pool=DummyPool())


def test_materialize_dataset_hadoop_config(synthetic_dataset):
    """Test that using materialize_dataset does not alter the hadoop_config"""
    spark = SparkSession.builder.getOrCreate()
    hadoop_config = spark.sparkContext._jsc.hadoopConfiguration()

    parquet_summary_metadata = False
    parquet_row_group_check = 100

    # Set the parquet summary giles and row group size check min
    hadoop_config.setBoolean('parquet.enable.summary-metadata', parquet_summary_metadata)
    hadoop_config.setInt('parquet.row-group.size.row.check.min', parquet_row_group_check)
    assert hadoop_config.get('parquet.enable.summary-metadata') == str(parquet_summary_metadata).lower()
    assert hadoop_config.get('parquet.row-group.size.row.check.min') == str(parquet_row_group_check)
    destination = synthetic_dataset.path + '_moved'
    create_test_dataset('file://{}'.format(destination), range(10), spark=spark)

    # Check that they are back to the original values after writing the dataset
    hadoop_config = spark.sparkContext._jsc.hadoopConfiguration()
    assert hadoop_config.get('parquet.enable.summary-metadata') == str(parquet_summary_metadata).lower()
    assert hadoop_config.get('parquet.row-group.size.row.check.min') == str(parquet_row_group_check)
    # Other options should return to being unset
    assert hadoop_config.get('parquet.block.size') is None
    assert hadoop_config.get('parquet.block.size.row.check.min') is None
    spark.sparkContext.stop()
    rmtree(destination)


def test_dataset_path_is_a_unicode(synthetic_dataset):
    """Just a bunch of read and compares of all values to the expected values using the different reader pools"""
    # Making sure unicode_in_p23 is a unicode both in python 2 and 3
    unicode_in_p23 = synthetic_dataset.url.encode().decode('utf-8')
    with Reader(unicode_in_p23, reader_pool=DummyPool()) as reader:
        next(reader)
