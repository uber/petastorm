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
import unittest

import pytest

from examples.hello_world.hello_world_dataset import generate_hello_world_dataset
from examples.hello_world.pyspark_hello_world import pyspark_hello_world
from examples.hello_world.pytorch_hello_world import pytorch_hello_world
from petastorm.reader import Reader
from petastorm.tests.conftest import SyntheticDataset


@pytest.fixture(scope="session")
def hello_world_dataset(tmpdir_factory):
    path = tmpdir_factory.mktemp("data").strpath
    url = 'file://' + path

    generate_hello_world_dataset(url)

    dataset = SyntheticDataset(url=url, path=path, data=None)

    # Generate a dataset
    assert os.path.exists(os.path.join(path, '_SUCCESS'))

    return dataset


def test_generate(hello_world_dataset):
    # Read from it using a plain reader
    with Reader(hello_world_dataset.url) as reader:
        all_samples = list(reader)
    assert all_samples


@unittest.skip('Some conflict between pytorch and parquet shared libraries results in occasional '
               'segfaults in this case.')
def test_pytorch_hello_world_example(hello_world_dataset):
    pytorch_hello_world(hello_world_dataset.url)

def test_pyspark_hello_world_example(hello_world_dataset):
    pyspark_hello_world(hello_world_dataset.url)
