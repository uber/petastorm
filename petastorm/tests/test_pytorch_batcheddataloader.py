import numpy as np
import pyarrow as pa
import pytest
# Must import pyarrow before torch. See: https://github.com/uber/petastorm/blob/master/docs/troubleshoot.rst
import torch

from petastorm import make_reader, TransformSpec, make_batch_reader
from petastorm.pytorch import BatchedDataLoader
from petastorm.tests.test_common import TestSchema

BATCHABLE_FIELDS = set(TestSchema.fields.values()) - \
    {TestSchema.matrix_nullable, TestSchema.string_array_nullable, TestSchema.decimal, TestSchema.partition_key,
     TestSchema.matrix_string, TestSchema.empty_matrix_string, TestSchema.integer_nullable}

# pylint: disable=unnecessary-lambda
MINIMAL_READER_FLAVOR_FACTORIES = [
    lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs),
]

# pylint: disable=unnecessary-lambda
ALL_READER_FLAVOR_FACTORIES = MINIMAL_READER_FLAVOR_FACTORIES + [
    lambda url, **kwargs: make_reader(url, reader_pool_type='thread', **kwargs),
    lambda url, **kwargs: make_reader(url, reader_pool_type='process', pyarrow_serialize=False, **kwargs),
    lambda url, **kwargs: make_reader(url, reader_pool_type='process', workers_count=1, pyarrow_serialize=True,
                                      **kwargs),
]


def _check_simple_reader(loader, expected_data, expected_fields):
    # Read a bunch of entries from the dataset and compare the data to reference
    def _type(v):
        return v.dtype if isinstance(v, np.ndarray) else type(v)

    def _unbatch(x):
        if isinstance(x, torch.Tensor):
            x_numpy = x.numpy()
            assert x_numpy.shape[0] == 1
            return x_numpy.squeeze(0)
        elif isinstance(x, list):
            return x[0]
        else:
            raise RuntimeError('Unexpected type while unbatching.')

    expected_field_names = [f.name for f in expected_fields]
    for actual in loader:
        actual_numpy = {k: _unbatch(v) for k, v in actual.items() if k in expected_field_names}
        expected_all_fields = next(d for d in expected_data if d['id'] == actual_numpy['id'])
        expected = {k: v for k, v in expected_all_fields.items() if k in expected_field_names}
        np.testing.assert_equal(actual_numpy, expected)
        actual_types = [_type(v) for v in actual_numpy.values()]
        expected_types = [_type(v) for v in actual_numpy.values()]
        assert actual_types == expected_types


def _sensor_name_to_int(row):
    result_row = dict(**row)
    result_row['sensor_name'] = 0
    return result_row


@pytest.mark.parametrize('reader_factory', ALL_READER_FLAVOR_FACTORIES)
def test_simple_read(synthetic_dataset, reader_factory):
    with BatchedDataLoader(reader_factory(synthetic_dataset.url, schema_fields=BATCHABLE_FIELDS,
                                          transform_spec=TransformSpec(_sensor_name_to_int))) as loader:
        _check_simple_reader(loader, synthetic_dataset.data, BATCHABLE_FIELDS - {TestSchema.sensor_name})


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_no_shuffling(synthetic_dataset, reader_factory):
    with BatchedDataLoader(reader_factory(synthetic_dataset.url, schema_fields=['^id$'], workers_count=1,
                                          shuffle_row_groups=False)) as loader:
        ids = [row['id'][0].numpy() for row in loader]
        # expected_ids would be [0, 1, 2, ...]
        expected_ids = [row['id'] for row in synthetic_dataset.data]
        np.testing.assert_array_equal(expected_ids, ids)


@pytest.mark.parametrize('reader_factory', MINIMAL_READER_FLAVOR_FACTORIES)
def test_with_shuffling_buffer(synthetic_dataset, reader_factory):
    with BatchedDataLoader(reader_factory(synthetic_dataset.url, schema_fields=['^id$'], workers_count=1,
                                          shuffle_row_groups=False),
                           shuffling_queue_capacity=51) as loader:
        ids = [row['id'][0].numpy() for row in loader]

        assert len(ids) == len(synthetic_dataset.data), 'All samples should be returned after reshuffling'

        # diff(ids) would return all-'1' for the seqeunce (note that we used shuffle_row_groups=False)
        # We assume we get less then 10% of consequent elements for the sake of the test (this probability is very
        # close to zero)
        assert sum(np.diff(ids) == 1) < len(synthetic_dataset.data) / 10.0


@pytest.mark.parametrize('shuffling_queue_capacity', [0, 2, 10, 1000])
def test_with_batch_reader(scalar_dataset, shuffling_queue_capacity):
    """See if we are getting correct batch sizes when using BatchedDataLoader with make_batch_reader"""
    pytorch_compatible_fields = [k for k, v in scalar_dataset.data[0].items()
                                 if not isinstance(v, (np.datetime64, np.unicode_))]
    with BatchedDataLoader(make_batch_reader(scalar_dataset.url, schema_fields=pytorch_compatible_fields),
                           batch_size=3, shuffling_queue_capacity=shuffling_queue_capacity) as loader:
        batches = list(loader)
        assert len(scalar_dataset.data) == sum(batch['id'].shape[0] for batch in batches)

        # list types are broken in pyarrow 0.15.0. Don't test list-of-int field
        if pa.__version__ != '0.15.0':
            assert len(scalar_dataset.data) == sum(batch['int_fixed_size_list'].shape[0] for batch in batches)
            assert batches[0]['int_fixed_size_list'].shape[1] == len(scalar_dataset.data[0]['int_fixed_size_list'])
