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

import pyarrow.parquet as pq
from petastorm.reader import Reader, ShuffleOptions


def test_dataset_url_must_be_string():
    with pytest.raises(ValueError):
        Reader(dataset_url=None)

    with pytest.raises(ValueError):
        Reader(dataset_url=123)

    with pytest.raises(ValueError):
        Reader(dataset_url=[])


def test_normalize_shuffle_partitions(synthetic_dataset):
    dataset = pq.ParquetDataset(synthetic_dataset.path)
    shuffle_options = ShuffleOptions(True, 2)
    Reader._normalize_shuffle_options(shuffle_options, dataset)
    assert shuffle_options.shuffle_row_drop_partitions == 2

    shuffle_options = ShuffleOptions(True, 1000)
    Reader._normalize_shuffle_options(shuffle_options, dataset)
    assert shuffle_options.shuffle_row_drop_partitions == 10
