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

import pytest

from petastorm import make_reader


def dataset_urls():
    """Returns a list of legacy datasets available for testing"""
    legacy_data_directory = os.path.join(os.path.dirname(__file__), 'data', 'legacy')
    versions = os.listdir(legacy_data_directory)
    urls = ['file://' + os.path.join(legacy_data_directory, v) for v in versions]
    return urls


@pytest.mark.parametrize('legacy_dataset_url', dataset_urls())
def test_reading_legacy_dataset(legacy_dataset_url):
    """The test runs for a single legacy dataset. Opens the dataset using `make_reader` and reads all records from it"""
    with make_reader(legacy_dataset_url, workers_count=1) as reader:
        all_data = list(reader)

        # Some basic check on the data
        assert len(all_data) == 100
        assert len(all_data[0]._fields) > 5
        assert all_data[0].matrix.shape == (32, 16, 3)
