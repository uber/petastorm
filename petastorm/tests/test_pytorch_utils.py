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
from __future__ import division

from petastorm.pytorch import DataLoader
from petastorm.reader import Reader
from petastorm.workers_pool.dummy_pool import DummyPool


def _noop_collate(alist):
    return alist


def test_basic_pytorch_dataloader(synthetic_dataset):
    loader = DataLoader(Reader(synthetic_dataset.url, reader_pool=DummyPool()), collate_fn=_noop_collate)
    for item in loader:
        assert len(item) == 1


def test_pytorch_dataloader_batched(synthetic_dataset):
    batch_size = 10
    loader = DataLoader(Reader(synthetic_dataset.url, reader_pool=DummyPool()),
                        batch_size=batch_size, collate_fn=_noop_collate)
    for item in loader:
        assert len(item) == batch_size


def test_pytorch_dataloader_context(synthetic_dataset):
    with DataLoader(Reader(synthetic_dataset.url, reader_pool=DummyPool()),
                    collate_fn=_noop_collate) as loader:
        for item in loader:
            assert len(item) == 1
