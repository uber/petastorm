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

import numpy as np
import pytest

from examples.mnist.generate_petastorm_mnist import download_mnist_data, \
    mnist_data_to_petastorm_dataset

# Set test image sizes and number of mock nouns/variants
MOCK_IMAGE_SIZE = (28, 28)
MOCK_IMAGE_3DIM_SIZE = (28, 28, 1)
MOCK_IMAGE_COUNT = 5


class MockDataObj(object):
    """ Wraps a mock image array and provide a needed getdata() interface function. """
    def __init__(self, a):
        self.a = a

    def getdata(self):
        return self.a


@pytest.fixture(scope="session")
def mock_mnist_data():
    """
    Creates a mock data dictionary with train and test sets, each containing 5 mock pairs:

        ``(random images, random digit)``.
    """
    bogus_data = {
        'train': [],
        'test': []
    }

    for dset, data in bogus_data.items():
        for i in range(MOCK_IMAGE_COUNT):
            pair = (MockDataObj(np.random.randint(0, 255, size=MOCK_IMAGE_SIZE, dtype=np.uint8)), np.random.randint(0, 9))
            data.append(pair)

    return bogus_data


def test_image_to_numpy(mock_mnist_data):
    """ Show output of image object reshaped as numpy array """
    im = mock_mnist_data['train'][0]
    print(im)
    print(im[1])
    assert im[1] >= 0 and im[1] <= 9

    print(im[0].getdata())
    assert im[0].getdata().shape == MOCK_IMAGE_SIZE

    np.set_printoptions(linewidth=200)
    reshaped = np.array(list(im[0].getdata()), dtype=np.uint8).reshape(MOCK_IMAGE_3DIM_SIZE)
    print(reshaped)
    assert reshaped.shape == MOCK_IMAGE_3DIM_SIZE


# This test requires torch import, but that's causing dlopen to fail in Travis run. :(
# So skipping it for now
def _dont_test_mnist_download(tmpdir):
    """ Demonstrates that MNIST download works, using only the 'test' data. Assumes data does not change often. """
    o = download_mnist_data(tmpdir, train=False)
    assert 10000 == len(o)
    assert o[0][1] == 7
    assert o[len(o)-1][1] == 6


def test_generate(mock_mnist_data, tmpdir):
    # Using parquet_files_count to speed up the test
    mnist_data_to_petastorm_dataset(tmpdir, 'file://{}'.format(tmpdir),
                                    spark_master='local[3]', parquet_files_count=1,
                                    mnist_data=mock_mnist_data)
