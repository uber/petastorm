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

import os
import shutil
import tempfile
import unittest

import numpy as np

from examples.mnist.generate_petastorm_mnist import download_mnist_data, \
    mnist_data_to_petastorm_dataset

# Set test image sizes and number of mock nouns/variants
MOCK_IMAGE_SIZE = (28, 28, 1)
MOCK_IMAGE_COUNT = 5


class MockDataObj(object):
    def __init__(self, a):
        self.a = a

    def getdata(self):
        return self.a


def _mock_mnist_data(temp_dir):
    """Creates a mock directory with 5 noun-id directores and 3 variants of the noun images. Random images are used."""
    bogus_data = {
        'train': [],
        'test': []
    }

    for dset, data in bogus_data.items():
        for i in range(MOCK_IMAGE_COUNT):
            pair = (MockDataObj(np.random.randint(0, 255, size=MOCK_IMAGE_SIZE, dtype=np.uint8)), np.random.randint(0, 255))
            data.append(pair)

    return bogus_data


class TestGenerate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mock_mnist_dir = tempfile.mkdtemp()
        cls.mock_output_dir = tempfile.mkdtemp()
        cls.mnist_data = _mock_mnist_data(cls.mock_mnist_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.mock_mnist_dir):
            shutil.rmtree(cls.mock_mnist_dir)
        if os.path.exists(cls.mock_output_dir):
            shutil.rmtree(cls.mock_output_dir)

    @unittest.skip('Demonstrates image conversion to numpty array')
    def test_image_to_numpy(self):
        """ Show output of image object reshaped as numpy array """
        im = self.mnist_data['train'][0]
        print(im)
        print(im[1])
        print(list(im[0].getdata()))
        np.set_printoptions(linewidth=200)
        print(np.array(list(im[0].getdata()), dtype=np.uint8).reshape(28, 28))

    def test_mnist_download(self):
        """ Demonstrates that MNIST download works, using only the 'test' data. """
        o = download_mnist_data(self.mock_mnist_dir, train=False)
        self.assertEqual(10000, len(o))
        self.assertEqual(o[0][1], 7)
        self.assertEqual(o[len(o)-1][1], 6)

    def test_generate(self):
        # Use parquet_files_count to speed up the test
        mnist_data_to_petastorm_dataset(self.mock_mnist_dir,
                                        'file://' + self.mock_output_dir,
                                        spark_master='local[3]', parquet_files_count=1,
                                        mnist_data=self.mnist_data)


if __name__ == '__main__':
    unittest.main()
