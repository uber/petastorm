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

from pyarrow import parquet as pq

from petastorm import utils


def test_make_parquet_dataset_drops_validate_schema_when_unsupported(monkeypatch):
    captured = {}

    class FakeParquetDataset(object):
        def __init__(self, *args, **kwargs):
            captured['kwargs'] = kwargs

    monkeypatch.setattr(pq, 'ParquetDataset', FakeParquetDataset)
    monkeypatch.setattr(utils, '_PARQUET_DATASET_SUPPORTS_VALIDATE_SCHEMA', False)

    utils.make_parquet_dataset('dataset-path', validate_schema=False, metadata_nthreads=10)

    assert 'validate_schema' not in captured['kwargs']
    assert captured['kwargs']['metadata_nthreads'] == 10
