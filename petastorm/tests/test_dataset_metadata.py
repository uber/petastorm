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

import pytest

from petastorm.etl.dataset_metadata import get_schema_from_dataset_url
from petastorm.tests.test_common import TestSchema


def test_get_schema_from_dataset_url(synthetic_dataset):
    schema = get_schema_from_dataset_url(synthetic_dataset.url)
    assert TestSchema.fields == schema.fields


def test_get_schema_from_dataset_url_bogus_url():
    with pytest.raises(IOError):
        get_schema_from_dataset_url('file:///non-existing-path')

    with pytest.raises(ValueError):
        get_schema_from_dataset_url('/invalid_url')
