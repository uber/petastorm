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

import numpy as np
import pytest
import torch  # pylint: disable=unused-import

import examples.mnist.pytorch_example as pytorch_example
import examples.mnist.tf_example as tf_example
from examples.mnist.generate_petastorm_mnist import download_mnist_data, \
    mnist_data_to_petastorm_dataset
from petastorm.reader import Reader
from petastorm.workers_pool.dummy_pool import DummyPool

logging.basicConfig(level=logging.INFO)

# Set test image sizes and number of mock nouns/variants
MOCK_IMAGE_SIZE = (28, 28)
MOCK_IMAGE_3DIM_SIZE = (28, 28, 1)
SMALL_MOCK_IMAGE_COUNT = {
    'train': 30,
    'test': 5
}
LARGE_MOCK_IMAGE_COUNT = {
    'train': 600,
    'test': 100
}


class MockDataObj(object):
    """ Wraps a mock image array and provide a needed getdata() interface function. """

    def __init__(self, a):
        self.a = a

    def getdata(self):
        return self.a


def _mock_mnist_data(mock_spec):
    """
    Creates a mock data dictionary with train and test sets, each containing 5 mock pairs:

        ``(random images, random digit)``.
    """
    bogus_data = {
        'train': [],
        'test': []
    }

    for dset, data in bogus_data.items():
        for _ in range(mock_spec[dset]):
            pair = (MockDataObj(np.random.randint(0, 255, size=MOCK_IMAGE_SIZE, dtype=np.uint8)),
                    np.random.randint(0, 9))
            data.append(pair)

    return bogus_data


@pytest.fixture(scope="session")
def small_mock_mnist_data():
    return _mock_mnist_data(SMALL_MOCK_IMAGE_COUNT)


@pytest.fixture(scope="session")
def large_mock_mnist_data():
    return _mock_mnist_data(LARGE_MOCK_IMAGE_COUNT)


@pytest.fixture(scope="session")
def generate_mnist_dataset(small_mock_mnist_data, tmpdir_factory):
    # Using parquet_files_count to speed up the test
    path = tmpdir_factory.mktemp('data').strpath
    dataset_url = 'file://{}'.format(path)
    mnist_data_to_petastorm_dataset(path, dataset_url, mnist_data=small_mock_mnist_data,
                                    spark_master='local[1]', parquet_files_count=1)
    return path


def test_image_to_numpy(small_mock_mnist_data):
    log = logging.getLogger('test_image_to_numpy')
    """ Show output of image object reshaped as numpy array """
    im = small_mock_mnist_data['train'][0]
    log.debug(im)
    log.debug(im[1])
    assert 0 <= im[1] <= 9

    log.debug(im[0].getdata())
    assert im[0].getdata().shape == MOCK_IMAGE_SIZE

    np.set_printoptions(linewidth=200)
    reshaped = np.array(list(im[0].getdata()), dtype=np.uint8).reshape(MOCK_IMAGE_3DIM_SIZE)
    log.debug(reshaped)
    assert reshaped.shape == MOCK_IMAGE_3DIM_SIZE


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
    assert os.path.exists(os.path.join(train_path, '_metadata'))

    test_path = os.path.join(generate_mnist_dataset, 'test')
    assert os.path.exists(test_path)
    assert os.path.exists(os.path.join(test_path, '_common_metadata'))
    assert os.path.exists(os.path.join(test_path, '_metadata'))


def test_read_mnist_dataset(generate_mnist_dataset):
    # Verify both datasets via a reader
    for dset in SMALL_MOCK_IMAGE_COUNT.keys():
        with Reader('file://{}/{}'.format(generate_mnist_dataset, dset), reader_pool=DummyPool()) as reader:
            assert len(reader) == SMALL_MOCK_IMAGE_COUNT[dset]


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

    with DataLoader(Reader('{}/train'.format(dataset_url), reader_pool=DummyPool(), num_epochs=1),
                    batch_size=32, transform=pytorch_example._transform_row) as train_loader:
        pytorch_example.train(model, device, train_loader, 10, optimizer, 1)
    with DataLoader(Reader('{}/test'.format(dataset_url), reader_pool=DummyPool(), num_epochs=1),
                    batch_size=100, transform=pytorch_example._transform_row) as test_loader:
        pytorch_example.test(model, device, test_loader)


def test_full_tf_example(large_mock_mnist_data, tmpdir):
    # First, generate mock dataset
    dataset_url = 'file://{}'.format(tmpdir)
    mnist_data_to_petastorm_dataset(tmpdir, dataset_url, mnist_data=large_mock_mnist_data,
                                    spark_master='local[1]', parquet_files_count=1)

    # Tensorflow train and test
    tf_example.train_and_test(
        dataset_url=dataset_url,
        training_iterations=10,
        batch_size=10,
        evaluation_interval=10,
    )
