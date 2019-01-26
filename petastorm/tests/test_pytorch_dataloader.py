from decimal import Decimal

import numpy as np
import pytest
# Must import pyarrow before torch. See: https://github.com/uber/petastorm/blob/master/docs/troubleshoot.rst
import pyarrow  # noqa: F401 pylint: disable=W0611
import torch

from petastorm import make_reader, TransformSpec
from petastorm.pytorch import _sanitize_pytorch_types, DataLoader, decimal_friendly_collate
from petastorm.tests.test_common import TestSchema

BATCHABLE_FIELDS = set(TestSchema.fields.values()) - \
                   {TestSchema.matrix_nullable, TestSchema.string_array_nullable,
                    TestSchema.matrix_string, TestSchema.empty_matrix_string}

# pylint: disable=unnecessary-lambda
MINIMAL_READER_FLAVOR_FACTORIES = [
    lambda url, **kwargs: make_reader(url, reader_pool_type='dummy', **kwargs),
    # Disabling v2 since it does not support transform_spec which is needed in reading data into pytorch
    # lambda url, **kwargs: make_reader(url, reader_engine='experimental_reader_v2', **kwargs)
]

# pylint: disable=unnecessary-lambda
ALL_READER_FLAVOR_FACTORIES = MINIMAL_READER_FLAVOR_FACTORIES + [
    lambda url, **kwargs: make_reader(url, reader_pool_type='thread', **kwargs),
    lambda url, **kwargs: make_reader(url, reader_pool_type='process', pyarrow_serialize=False, **kwargs),
    lambda url, **kwargs: make_reader(url, reader_pool_type='process', workers_count=1, pyarrow_serialize=True,
                                      **kwargs),
    # Disabling v2 since it does not support transform_spec which is needed in reading data into pytorch
    # lambda url, **kwargs: make_reader(url, reader_engine='experimental_reader_v2', reader_pool_type='process',
    # **kwargs)
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
    with DataLoader(reader_factory(synthetic_dataset.url, schema_fields=BATCHABLE_FIELDS,
                                   transform_spec=TransformSpec(_sensor_name_to_int))) as loader:
        _check_simple_reader(loader, synthetic_dataset.data, BATCHABLE_FIELDS - {TestSchema.sensor_name})


def test_sanitize_pytorch_types_int8():
    dict_to_sanitize = {'a': np.asarray([-1, 1], dtype=np.int8)}
    _sanitize_pytorch_types(dict_to_sanitize)
    np.testing.assert_array_equal(dict_to_sanitize['a'], [-1, 1])
    assert dict_to_sanitize['a'].dtype == np.int16


def test_decimal_friendly_collate_empty_input():
    assert decimal_friendly_collate([dict()]) == dict()


@pytest.mark.parametrize('numpy_dtype',
                         [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64])
def test_torch_tensorable_types(numpy_dtype):
    """Make sure that we 'sanitize' only integer types that can not be made into torch tensors natively"""
    value = np.zeros((2, 2), dtype=numpy_dtype)
    dict_to_sanitize = {'value': value}
    _sanitize_pytorch_types(dict_to_sanitize)

    torchable = False
    try:
        torch.Tensor(value)
        torchable = True
    except TypeError:
        pass

    tensor = torch.as_tensor(dict_to_sanitize['value'])

    tensor_and_back = tensor.numpy()

    if tensor_and_back.dtype != value.dtype:
        assert tensor_and_back.dtype.itemsize > value.dtype.itemsize
        assert not torchable, '_sanitize_pytorch_types modified value of type {}, but it was possible to create a ' \
                              'Tensor directly from a value with that type'.format(numpy_dtype)


def test_decimal_friendly_collate_input_has_decimals_in_dictionary():
    desired = {
        'decimal': [Decimal('1.0'), Decimal('1.1')],
        'int': [1, 2]
    }
    input_batch = [
        {'decimal': Decimal('1.0'), 'int': 1},
        {'decimal': Decimal('1.1'), 'int': 2},
    ]
    actual = decimal_friendly_collate(input_batch)

    assert len(actual) == 2
    assert desired['decimal'] == actual['decimal']
    np.testing.assert_equal(desired['int'], actual['int'].numpy())


def test_decimal_friendly_collate_input_has_decimals_in_tuple():
    input_batch = ([Decimal('1.0'), 1], [Decimal('1.1'), 2])
    desired = [(Decimal('1.0'), Decimal('1.1')), (1, 2)]
    actual = decimal_friendly_collate(input_batch)

    assert len(actual) == 2
    assert desired[0] == actual[0]
    np.testing.assert_equal(desired[1], actual[1].numpy())
