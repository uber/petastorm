#
# Uber, Inc. (c) 2017
#
import json
import os
import random
import unittest
from shutil import rmtree, copytree
from tempfile import mkdtemp

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pyarrow import parquet as pq
from pyspark.sql.types import LongType, ShortType, StringType

from dataset_toolkit.local_disk_cache import LocalDiskCache
from dataset_toolkit.codecs import ScalarCodec
from dataset_toolkit.etl.dataset_metadata import ROW_GROUPS_PER_FILE_KEY, \
    ROW_GROUPS_PER_FILE_KEY_ABSOLUTE_PATHS
from dataset_toolkit.reader import Reader
from dataset_toolkit.tests.tempdir import temporary_directory
from dataset_toolkit.selectors import SingleIndexSelector
from dataset_toolkit.tests.test_common import create_test_dataset, TestSchema
from dataset_toolkit.tests.test_end_to_end_predicates_impl import \
    PartitionKeyInSetPredicate, EqualPredicate
from dataset_toolkit.unischema import UnischemaField, Unischema
from dataset_toolkit.workers_pool.dummy_pool import DummyPool
from dataset_toolkit.workers_pool.process_pool import ProcessPool
from dataset_toolkit.workers_pool.thread_pool import ThreadPool

# Number of rows in a fake dataset

ROWS_COUNT = 1000


class EndToEndDatasetToolkitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes dataset once per test. All tests in this class will use the same fake dataset."""
        # Write a fake dataset to this location
        cls._dataset_dir = mkdtemp('end_to_end_dataset_toolkit')
        cls._dataset_url = 'file://{}'.format(cls._dataset_dir)
        cls._dataset_dicts = create_test_dataset(cls._dataset_url, range(ROWS_COUNT))

    @classmethod
    def tearDownClass(cls):
        # Remove everything created with "get_temp_dir"
        rmtree(cls._dataset_dir)

    def _check_simple_reader(self, reader):
        # Read a bunch of entries from the dataset and compare the data to reference
        for row in reader:
            actual = dict(row._asdict())
            expected = next(d for d in self._dataset_dicts if d['id'] == actual['id'])
            np.testing.assert_equal(expected, actual)

    def test_simple_read(self):
        """Just a bunch of read and compares of all values to the expected values using the different reader pools"""
        pool_impls = [DummyPool, ThreadPool, ProcessPool]
        for pool_impl in pool_impls:
            with Reader(dataset_url=self._dataset_url, reader_pool=pool_impl(10)) as reader:
                self._check_simple_reader(reader)

    def test_simple_read_with_disk_cache(self):
        """Try using the Reader with LocalDiskCache using different flavors of pools"""
        pool_impls = [DummyPool, ThreadPool, ProcessPool]
        for pool_impl in pool_impls:
            with temporary_directory() as cache_dir:
                CACHE_SIZE = 10 * 2 ** 30  # 20GB
                ROW_SIZE_BYTES = 100  # not really important for this test
                with Reader(dataset_url=self._dataset_url, reader_pool=pool_impl(10), num_epochs=2,
                            cache=LocalDiskCache(cache_dir, CACHE_SIZE, ROW_SIZE_BYTES)) as reader:
                    self._check_simple_reader(reader)

    def test_simple_read_with_added_slashes(self):
        """Tests that using relative paths for the dataset metadata works as expected"""
        with Reader(dataset_url=self._dataset_url + '///', reader_pool=DummyPool()) as reader:
            self._check_simple_reader(reader)

    def test_simple_read_moved_dataset(self):
        """Tests that a dataset may be opened after being moved to a new location"""
        destination = self._dataset_dir + '_moved'
        copytree(self._dataset_dir, destination)

        with Reader(dataset_url='file://{}'.format(destination), reader_pool=DummyPool()) as reader:
            self._check_simple_reader(reader)

        rmtree(destination)

    def test_simple_read_absolute_paths_metadata(self):
        """Tests that a dataset using absolute paths in the dataset toolkit metadata may still be opened"""

        # Create a copied version of the dataset and change the metadata to use absolute paths
        destination = self._dataset_dir + '_backwards_compatible'
        copytree(self._dataset_dir, destination)

        dataset = pq.ParquetDataset(destination, validate_schema=False)
        old_row_group_nums = json.loads(dataset.common_metadata.metadata[ROW_GROUPS_PER_FILE_KEY])
        row_group_nums = {}
        for rel_path, num_row_groups in old_row_group_nums.iteritems():
            row_group_nums[os.path.join(dataset.paths, rel_path)] = num_row_groups
        base_schema = dataset.common_metadata.schema.to_arrow_schema()
        metadata_dict = base_schema.metadata
        del metadata_dict[ROW_GROUPS_PER_FILE_KEY]
        metadata_dict[ROW_GROUPS_PER_FILE_KEY_ABSOLUTE_PATHS] = json.dumps(row_group_nums)
        schema = base_schema.add_metadata(metadata_dict)
        with dataset.fs.open(os.path.join(destination, '_metadata'), 'wb') as metadata_file:
            pq.write_metadata(schema, metadata_file)

        with Reader(dataset_url='file://{}'.format(destination), reader_pool=DummyPool()) as reader:
            self._check_simple_reader(reader)

        rmtree(destination)

    def test_reading_subset_of_columns(self):
        """Just a bunch of read and compares of all values to the expected values"""
        with Reader(schema_fields=[TestSchema.id2, TestSchema.id],
                    dataset_url=self._dataset_url,
                    reader_pool=DummyPool()) as reader:
            # Read a bunch of entries from the dataset and compare the data to reference
            for row in reader:
                actual = dict(row._asdict())
                expected = next(d for d in self._dataset_dicts if d['id'] == actual['id'])
                np.testing.assert_equal(expected['id2'], actual['id2'])

    def test_shuffle(self):
        rows_count = len(self._dataset_dicts)

        def readout_all_ids(shuffle):
            with Reader(dataset_url=self._dataset_url, reader_pool=ThreadPool(1), shuffle=shuffle) as reader:
                ids = [row.id for row in reader]
            return ids

        # Read ids twice without shuffle: assert we have the same array and all expected ids are in the array
        first_readout = readout_all_ids(False)
        second_readout = readout_all_ids(False)
        np.testing.assert_array_equal(range(rows_count), sorted(first_readout))
        np.testing.assert_array_equal(first_readout, second_readout)

        # Now read with shuffling
        shuffled_readout = readout_all_ids(True)
        self.assertTrue(np.any(np.not_equal(first_readout, shuffled_readout)))

    def test_predicate_on_partition(self):
        for expected_partition_keys in [{'p_0', 'p_2'}, {'p_0'}, {'p_1', 'p_2'}]:
            for pool in [ProcessPool, ThreadPool, DummyPool]:
                with Reader(dataset_url=self._dataset_url,
                            shuffle=True,
                            reader_pool=pool(10),
                            predicate=PartitionKeyInSetPredicate(expected_partition_keys)) as reader:
                    partition_keys = set(row.partition_key for row in reader)
                    self.assertEqual(partition_keys, expected_partition_keys)

    def test_predicate_on_multiple_fields(self):
        for pool in [ProcessPool, ThreadPool, DummyPool]:
            expected_values = {'id': 11, 'id2': 1}
            with Reader(dataset_url=self._dataset_url, shuffle=False,
                        reader_pool=pool(10),
                        predicate=EqualPredicate(expected_values)) as reader:
                actual = next(reader)
                self.assertEqual(actual.id, expected_values['id'])
                self.assertEqual(actual.id2, expected_values['id2'])

    def test_predicate_with_invalid_fields(self):
        """Try passing an invalid field name from a predicate to the reader. An error should be raised."""
        TEST_CASES = [
            {'invalid_field_name': 1},
            dict(),
            {'invalid_field_name': 1, 'id': 11},
            {'invalid_field_name': 1, 'invalid_field_name_2': 11}]

        for predicate_spec in TEST_CASES:
            with Reader(dataset_url=self._dataset_url, shuffle=False,
                        reader_pool=ThreadPool(1),
                        predicate=EqualPredicate(predicate_spec)) as reader:
                with self.assertRaises(ValueError):
                    next(reader)

    def test_partition_multi_node(self):
        """Tests that the reader only returns half of the expected data consistently"""
        reader = Reader(dataset_url=self._dataset_url, reader_pool=DummyPool(),
                        training_partition=0, num_training_partitions=5)
        reader_2 = Reader(dataset_url=self._dataset_url, reader_pool=DummyPool(),
                          training_partition=0, num_training_partitions=5)

        results_1 = []
        expected = []
        for row in reader:
            actual = dict(row._asdict())
            results_1.append(actual)
            expected.append(next(d for d in self._dataset_dicts if d['id'] == actual['id']))

        results_2 = [dict(row._asdict()) for row in reader_2]

        # Since order is non deterministic, we need to sort results by id
        results_1.sort(key=lambda x: x['id'])
        results_2.sort(key=lambda x: x['id'])
        expected.sort(key=lambda x: x['id'])

        np.testing.assert_equal(expected, results_1)
        np.testing.assert_equal(results_1, results_2)

        assert len(results_1) < len(self._dataset_dicts)

        # Test that separate partitions also have no overlap by checking ids
        id_set = set([item['id'] for item in results_1])
        for partition in range(1, 5):
            with Reader(dataset_url=self._dataset_url, reader_pool=DummyPool(),
                        training_partition=partition, num_training_partitions=5) as reader_other:

                for row in reader_other:
                    self.assertTrue(dict(row._asdict())['id'] not in id_set)

        reader.stop()
        reader.join()
        reader_2.stop()
        reader_2.join()

    def test_partition_value_error(self):
        """Tests that the reader raises value errors when appropriate"""

        with self.assertRaises(ValueError):
            Reader(dataset_url=self._dataset_url, reader_pool=DummyPool(),
                   training_partition=0)

        with self.assertRaises(ValueError):
            Reader(dataset_url=self._dataset_url, reader_pool=DummyPool(),
                   num_training_partitions=5)

        with self.assertRaises(ValueError):
            Reader(dataset_url=self._dataset_url, reader_pool=DummyPool(),
                   training_partition='0', num_training_partitions=5)

        with self.assertRaises(ValueError):
            Reader(dataset_url=self._dataset_url, reader_pool=DummyPool(),
                   training_partition=0, num_training_partitions='5')

    def test_stable_pieces_order(self):
        """Tests that the reader raises value errors when appropriate"""

        RERUN_THE_TEST_COUNT = 20
        baseline_run = None
        for _ in range(RERUN_THE_TEST_COUNT):
            with Reader(schema_fields=[TestSchema.id],
                        dataset_url=self._dataset_url,
                        reader_pool=DummyPool(),
                        shuffle=False) as reader:
                this_run = [row.id for row in reader]
            if baseline_run:
                self.assertEqual(this_run, baseline_run)

            baseline_run = this_run

    def test_invalid_schema_field(self):
        # Let's assume we are selecting columns using a schema which is different from the one
        # stored in the dataset. Would expect to get a reasonable error message
        BogusSchema = Unischema('BogusSchema', [
            UnischemaField('partition_key', np.string_, (), ScalarCodec(StringType()), False),
            UnischemaField('id', np.int64, (), ScalarCodec(LongType()), False),
            UnischemaField('bogus_key', np.int32, (), ScalarCodec(ShortType()), False)])

        expected_values = {'bogus_key': 11, 'id': 1}
        with self.assertRaises(ValueError) as e:
            Reader(schema_fields=BogusSchema.fields.values(),
                   dataset_url=self._dataset_url, shuffle=False,
                   reader_pool=ThreadPool(1),
                   predicate=EqualPredicate(expected_values))

        self.assertTrue('bogus_key' in e.exception.message)

    def test_single_column_predicate(self):
        """Test quering a single column with a predicate on the same column """
        with Reader(schema_fields=[TestSchema.id],
                    dataset_url=self._dataset_url,
                    reader_pool=ThreadPool(1),
                    predicate=EqualPredicate({'id': 1})) as reader:
            # Read a bunch of entries from the dataset and compare the data to reference
            for row in reader:
                actual = dict(row._asdict())
                expected = next(d for d in self._dataset_dicts if d['id'] == actual['id'])
                np.testing.assert_equal(expected['id'], actual['id'])

    def test_multiple_epochs(self):
        """Tests that multiple epochs works as expected"""
        num_epochs = 5
        with Reader(
                dataset_url=self._dataset_url,
                reader_pool=DummyPool(),
                num_epochs=num_epochs) as reader:

            # Read all expected entries from the dataset and compare the data to reference
            id_set = set([d['id'] for d in self._dataset_dicts])

            for _ in range(num_epochs):
                current_epoch_set = set()
                for _ in range(len(id_set)):
                    actual = dict(next(reader)._asdict())
                    expected = next(d for d in self._dataset_dicts if d['id'] == actual['id'])
                    np.testing.assert_equal(expected, actual)
                    current_epoch_set.add(actual['id'])
                np.testing.assert_equal(id_set, current_epoch_set)

    def test_unlimited_epochs(self):
        """Tests that unlimited epochs works as expected"""
        with Reader(dataset_url=self._dataset_url,
                    reader_pool=DummyPool(),
                    num_epochs=None) as reader:

            # Read many expected entries from the dataset and compare the data to reference
            for _ in xrange(len(self._dataset_dicts) * random.randint(10, 30) + random.randint(25, 50)):
                actual = dict(next(reader)._asdict())
                expected = next(d for d in self._dataset_dicts if d['id'] == actual['id'])
                np.testing.assert_equal(expected, actual)

    def test_num_epochs_value_error(self):
        """Tests that the reader raises value errors when appropriate"""

        with self.assertRaises(ValueError):
            Reader(dataset_url=self._dataset_dir, reader_pool=DummyPool(),
                   num_epochs=0)

        with self.assertRaises(ValueError):
            Reader(dataset_url=self._dataset_dir, reader_pool=DummyPool(),
                   num_epochs=-10)

        with self.assertRaises(ValueError):
            Reader(dataset_url=self._dataset_dir, reader_pool=DummyPool(),
                   num_epochs='abc')

    def test_rowgroup_selector_integer_field(self):
        """ Select row groups to read based on dataset index for integer field"""
        with Reader(dataset_url=self._dataset_url,
                    rowgroup_selector=SingleIndexSelector(TestSchema.id.name, [2, 200]),
                    reader_pool=DummyPool()) as reader:
            status = [False, False]
            count = 0
            for row in reader:
                if row.id == 2:
                    status[0] = True
                if row.id == 200:
                    status[1] = True
                count += 1
            # both id values in reader result
            self.assertTrue(all(status))
            # read only 2 row groups, 100 rows per row group
            self.assertEqual(200, count)

    def test_rowgroup_selector_string_field(self):
        """ Select row groups to read based on dataset index for string field"""
        with Reader(dataset_url=self._dataset_url,
                    rowgroup_selector=SingleIndexSelector(TestSchema.sensor_name.name, ['test_sensor']),
                    reader_pool=DummyPool()) as reader:
            count = 0
            for row in reader:
                count += 1
            # Since we use artificial dataset all sensors have the same name,
            # so all row groups should be selected and all 1000 generated rows should be returned
            self.assertEqual(1000, count)

    def test_rowgroup_selector_wrong_index_name(self):
        """ Attempt to select row groups to based on wrong dataset index,
            Reader should raise exception
        """
        with self.assertRaises(ValueError):
            Reader(dataset_url=self._dataset_url,
                   rowgroup_selector=SingleIndexSelector('WrongIndexName', ['some_value']),
                   reader_pool=DummyPool())


if __name__ == '__main__':
    # Delegate to the test framework.
    unittest.main()
