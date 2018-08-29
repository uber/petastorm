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

from examples.hello_world.hello_world_dataset import generate_hello_world_dataset
from petastorm.reader import Reader


def test_generate(tmpdir):
    temp_url = 'file://' + tmpdir.strpath

    # Generate a dataset
    generate_hello_world_dataset(temp_url)
    assert '_SUCCESS' in os.listdir(tmpdir.strpath)

    # Read from it
    with Reader(temp_url) as reader:
        all_samples = list(reader)
    assert all_samples
