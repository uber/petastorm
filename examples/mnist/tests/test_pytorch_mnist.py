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

import logging
import os
import unittest

import pyarrow  # noqa: F401 pylint: disable=W0611
import torch

import pytest

import examples.mnist.pytorch_example as pytorch_example
from examples.mnist.generate_petastorm_mnist import mnist_data_to_petastorm_dataset, download_mnist_data
from examples.mnist.tests.conftest import SMALL_MOCK_IMAGE_COUNT
from petastorm import make_reader, TransformSpec

logging.basicConfig(level=logging.INFO)


# Set test image sizes and number of mock nouns/variants

@pytest.fixture(scope="session")
def generate_mnist_dataset(small_mock_mnist_data, tmpdir_factory):
    # Using parquet_files_count to speed up the test
    path = tmpdir_factory.mktemp('data').strpath
    dataset_url = 'file://{}'.format(path)
    mnist_data_to_petastorm_dataset(path, dataset_url, mnist_data=small_mock_mnist_data,
                                    spark_master='local[1]', parquet_files_count=1)
    return path


def test_full_pytorch_example(large_mock_mnist_data, tmpdir):
    # First, generate mock dataset
    dataset_url = 'file://{}'.format(tmpdir)
    mnist_data_to_petastorm_dataset(tmpdir, dataset_url, mnist_data=large_mock_mnist_data,
                                    spark_master='local[1]', parquet_files_count=1)

    # Next, run a round of training using the pytorce adapting data loader
    from petastorm.pytorch import DataLoader

    torch.manual_seed(1)
    device = torch.device('cpu')
    model = pytorch_example.Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    transform = TransformSpec(pytorch_example._transform_row, removed_fields=['idx'])

    with DataLoader(make_reader('{}/train'.format(dataset_url), reader_pool_type='dummy', num_epochs=1,
                                transform_spec=transform), batch_size=32) as train_loader:
        pytorch_example.train(model, device, train_loader, 10, optimizer, 1)
    with DataLoader(make_reader('{}/test'.format(dataset_url), reader_pool_type='dummy', num_epochs=1,
                                transform_spec=transform), batch_size=100) as test_loader:
        pytorch_example.test(model, device, test_loader)


@unittest.skip("Skipping this test since the server where the files are downloaded from is not stable")
def test_mnist_download(tmpdir):
    """ Demonstrates that MNIST download works, using only the 'test' data. Assumes data does not change often. """
    o = download_mnist_data(tmpdir, train=False)
    assert 10000 == len(o)
    assert o[0][1] == 7
    assert o[len(o) - 1][1] == 6


def test_generate_mnist_dataset(generate_mnist_dataset):
    train_path = os.path.join(generate_mnist_dataset, 'train')
    assert os.path.exists(train_path)
    assert os.path.exists(os.path.join(train_path, '_common_metadata'))

    test_path = os.path.join(generate_mnist_dataset, 'test')
    assert os.path.exists(test_path)
    assert os.path.exists(os.path.join(test_path, '_common_metadata'))


def test_read_mnist_dataset(generate_mnist_dataset):
    # Verify both datasets via a reader
    for dset in SMALL_MOCK_IMAGE_COUNT.keys():
        with make_reader('file://{}/{}'.format(generate_mnist_dataset, dset),
                         reader_pool_type='dummy', num_epochs=1) as reader:
            assert sum(1 for _ in reader) == SMALL_MOCK_IMAGE_COUNT[dset]
