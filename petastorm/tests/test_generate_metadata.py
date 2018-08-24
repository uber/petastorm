import os
from collections import namedtuple
from shutil import copytree

import pytest

import pyarrow.parquet as pq

from petastorm.reader import Reader
from petastorm.selectors import SingleIndexSelector
from petastorm.tests.test_common import create_test_dataset, TestSchema
from petastorm.workers_pool.dummy_pool import DummyPool
from petastorm.etl import petastorm_generate_metadata

ROWS_COUNT = 1000

SyntheticDataset = namedtuple('synthetic_dataset', ['url', 'data', 'path'])


@pytest.fixture(scope="session")
def synthetic_dataset(tmpdir_factory):
    path = tmpdir_factory.mktemp('data').strpath
    url = 'file://' + path
    data = create_test_dataset(url, range(ROWS_COUNT))
    return SyntheticDataset(url=url, path=path, data=data)


def _check_reader(path, rowgroup_selector=None):
    # Just check that you can open and read from a reader successfully
    with Reader('file://{}'.format(path), reader_pool=DummyPool(), rowgroup_selector=rowgroup_selector) as reader:
        [next(reader) for _ in range(10)]


def test_regenerate_row_group_metadata(synthetic_dataset, tmpdir):
    a_moved_path = tmpdir.join('moved').strpath
    copytree(synthetic_dataset.path, a_moved_path)

    # Make sure we can read dataset before
    _check_reader(a_moved_path)

    # Delete only the metadata file
    dataset = pq.ParquetDataset(a_moved_path)
    os.remove(dataset.metadata_path)

    # Should now raise a value error
    with pytest.raises(ValueError):
        _check_reader(a_moved_path)

    # Regenerate the metadata (taking the schema information from the common_metadata which exists)
    petastorm_generate_metadata._main(['--dataset_url', 'file://{}'.format(a_moved_path)])

    # Reader should now work again with rowgroup selector since it was in original metadata
    _check_reader(a_moved_path, SingleIndexSelector(TestSchema.id.name, [2, 18]))


def test_regenerate_all_metadata(synthetic_dataset, tmpdir):
    a_moved_path = tmpdir.join('moved').strpath
    copytree(synthetic_dataset.path, a_moved_path)

    # Make sure we can read dataset before
    _check_reader(a_moved_path)

    # Delete both metadata files
    dataset = pq.ParquetDataset(a_moved_path)
    os.remove(dataset.metadata_path)
    os.remove(dataset.common_metadata_path)

    # Should now raise a value error
    with pytest.raises(ValueError):
        _check_reader(a_moved_path)

    # Regenerate all metadata including unischema information
    petastorm_generate_metadata._main([
        '--dataset_url', 'file://{}'.format(a_moved_path),
        '--unischema_class', 'petastorm.tests.test_common.TestSchema',
    ])

    # Reader should now work again (row group selector will not since we removed all metadata)
    _check_reader(a_moved_path)


def test_cannot_find_unischema(synthetic_dataset, tmpdir):
    a_moved_path = tmpdir.join('moved').strpath
    copytree(synthetic_dataset.path, a_moved_path)

    # Make sure we can read dataset before
    _check_reader(a_moved_path)

    # Delete both metadata files
    dataset = pq.ParquetDataset(a_moved_path)
    os.remove(dataset.metadata_path)
    os.remove(dataset.common_metadata_path)

    # Should now raise a value error
    with pytest.raises(ValueError):
        _check_reader(a_moved_path)

    # Regeneration should fail since it cannot find the unischema
    with pytest.raises(ValueError):
        petastorm_generate_metadata._main(['--dataset_url', 'file://{}'.format(a_moved_path)])
