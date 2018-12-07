import os
from collections import namedtuple
from shutil import copytree

import pyarrow.parquet as pq
import pytest

from petastorm import make_reader
from petastorm.etl import petastorm_generate_metadata
from petastorm.selectors import SingleIndexSelector
from petastorm.tests.test_common import create_test_dataset, TestSchema

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
    with make_reader('file://{}'.format(path), reader_pool_type='dummy', rowgroup_selector=rowgroup_selector) as reader:
        [next(reader) for _ in range(10)]


def test_regenerate_metadata(synthetic_dataset, tmpdir):
    a_moved_path = tmpdir.join('moved').strpath
    copytree(synthetic_dataset.path, a_moved_path)

    # Make sure we can read dataset before
    _check_reader(a_moved_path)

    # Delete both metadata files
    dataset = pq.ParquetDataset(a_moved_path)
    os.remove(dataset.common_metadata_path)

    # make_reader should not be able to read a dataset without Petastorm metadat.
    with pytest.raises(RuntimeError, match='make_reader supports reading only Petastorm datasets'):
        _check_reader(a_moved_path)

    # Regenerate all metadata including unischema information
    petastorm_generate_metadata._main([
        '--dataset_url', 'file://{}'.format(a_moved_path),
        '--unischema_class', 'petastorm.tests.test_common.TestSchema',
    ])

    # Reader should now work again (row group selector will not since we removed all metadata)
    _check_reader(a_moved_path)


def test_regenerate_using_row_group_summary_metadata(synthetic_dataset, tmpdir):
    a_moved_path = tmpdir.join('moved').strpath
    copytree(synthetic_dataset.path, a_moved_path)

    # Make sure we can read dataset before
    _check_reader(a_moved_path)

    # Regenerate the metadata (taking the schema information from the common_metadata which exists)
    petastorm_generate_metadata._main(['--dataset_url', 'file://{}'.format(a_moved_path), '--use-summary-metadata'])

    dataset = pq.ParquetDataset(a_moved_path)
    # Metadata path should not exist still (should be only _common_metadata)
    assert dataset.metadata

    # Reader should now work again with rowgroup selector since it was in original metadata
    _check_reader(a_moved_path, SingleIndexSelector(TestSchema.id.name, [2, 18]))
