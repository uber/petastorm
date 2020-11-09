#  Copyright (c) 2017-2020 Uber Technologies, Inc.
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

from __future__ import print_function

import unittest
from functools import partial
from shutil import rmtree
from tempfile import mkdtemp

import pytest
import torch

from petastorm.pytorch import BatchedDataLoader, make_batched_reader_and_loader
from petastorm.reader import make_batch_reader, make_reader
from petastorm.tests.test_common import create_many_columns_non_petastorm_dataset

MEMORY_CACHE = 'mem_cache'


class ReaderLoaderWithMemoryCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes dataset once per test. All tests in this class will use the same fake dataset."""
        # Write a fake dataset to this location
        cls._num_rows = 50
        cls._dataset_dir = mkdtemp('test_metadata_read')
        cls._dataset_url = 'file://{}'.format(cls._dataset_dir)
        cls._dataset_dicts = create_many_columns_non_petastorm_dataset(cls._dataset_url,
                                                                       cls._num_rows,
                                                                       num_columns=1, num_files=4)

    @classmethod
    def tearDownClass(cls):
        """ Remove everything created in setUpClass. """
        rmtree(cls._dataset_dir)

    def test_mem_cache_reader_num_epochs_error(self):
        error_string = "When cache in loader memory is activated, reader.num_epochs_to_read " \
                       "must be set to 1. When caching the data in memory, we need to read " \
                       "the data only once. The rest of the time, we serve it from memory."
        with make_batch_reader(self._dataset_url,
                               num_epochs=2) as reader:
            with pytest.raises(ValueError, match=error_string):
                BatchedDataLoader(reader, cache_in_loader_memory=True, cache_size_limit=100,
                                  cache_row_size_estimate=1)

    def test_mem_cache_reader_cache_enabled_error(self):
        error_string = "num_epochs_to_load needs to be specified when cache_in_loader_memory is " \
                       "enabled."
        with make_batch_reader(self._dataset_url,
                               num_epochs=2) as reader:
            with pytest.raises(ValueError, match=error_string):
                BatchedDataLoader(reader, num_epochs_to_load=2)

    def test_mem_size_not_specified_error(self):
        error_string = 'Cannot create a in-memory cache. cache_size_limit and ' \
                       'cache_row_size_estimate must be larger than zero.'
        with make_batch_reader(self._dataset_url,
                               num_epochs=1) as reader:
            with pytest.raises(ValueError, match=error_string):
                BatchedDataLoader(reader, cache_in_loader_memory=True, cache_size_limit=100,
                                  cache_row_size_estimate=0)
            with pytest.raises(ValueError, match=error_string):
                BatchedDataLoader(reader, cache_in_loader_memory=True, cache_size_limit=0,
                                  cache_row_size_estimate=1)
            with pytest.raises(ValueError, match=error_string):
                BatchedDataLoader(reader, cache_in_loader_memory=True)

    def test_shuffling_q_size_error(self):
        error_string = "When using in-memory cache, shuffling_queue_capacity has no effect."
        with make_batch_reader(self._dataset_url,
                               num_epochs=1) as reader:
            with pytest.raises(ValueError, match=error_string):
                BatchedDataLoader(reader, cache_in_loader_memory=True, cache_size_limit=100,
                                  cache_row_size_estimate=1, shuffling_queue_capacity=100)

    def test_in_memory_cache_two_epoch(self):
        batch_size = 10
        for reader_factory in [make_reader, make_batch_reader]:
            for cache_type in [MEMORY_CACHE, None]:
                for shuffling_queue_capacity in [0, 20]:
                    print("testing reader_factor: {}, cache_type: {}, shuffling_queue_capacity: {}"
                          .format(reader_factory.__name__,
                                  cache_type,
                                  shuffling_queue_capacity))

                    if cache_type == MEMORY_CACHE:
                        extra_loader_params = dict(cache_size_limit=1000,
                                                   cache_row_size_estimate=1,
                                                   cache_in_loader_memory=True,
                                                   num_epochs_to_load=2)
                        extra_reader_params = dict(num_epochs=1)
                        print("extra_reader_params", extra_reader_params)
                    else:
                        extra_loader_params = \
                            dict(shuffling_queue_capacity=shuffling_queue_capacity)
                        extra_reader_params = dict(num_epochs=2)

                    with reader_factory(self._dataset_url,
                                        cur_shard=0,
                                        shard_count=1,
                                        reader_pool_type='thread',
                                        workers_count=2,
                                        hdfs_driver='libhdfs',
                                        schema_fields=['col_0'],
                                        **extra_reader_params) as reader:

                        loader = BatchedDataLoader(reader,
                                                   batch_size=batch_size,
                                                   transform_fn=partial(torch.as_tensor,
                                                                        device='cpu'),
                                                   **extra_loader_params)

                        it = iter(loader)
                        retrieved_so_far = None

                        for _ in range(5):
                            batch = next(it)
                            this_batch = batch['col_0'].clone()
                            assert list(this_batch.shape)[0] == batch_size

                            if retrieved_so_far is None:
                                retrieved_so_far = this_batch
                            else:
                                if cache_type == MEMORY_CACHE:
                                    intersect = set(retrieved_so_far.tolist()). \
                                        intersection(set(this_batch.tolist()))
                                    assert not intersect
                                retrieved_so_far = torch.cat([retrieved_so_far, this_batch], 0)

                        if cache_type == MEMORY_CACHE:
                            assert len(set(retrieved_so_far.tolist())) == self._num_rows

                        for _ in range(5):
                            batch = next(it)
                            if cache_type == MEMORY_CACHE:
                                assert loader._shuffling_buffer._done_adding
                            this_batch = batch['col_0'].clone()
                            assert list(this_batch.shape)[0] == batch_size
                            retrieved_so_far = torch.cat([retrieved_so_far, this_batch], 0)

                        with pytest.raises(StopIteration):
                            next(it)

    def test_in_memory_cache_two_epoch_with_make_fn(self):
        batch_size = 10
        for cache_type in [MEMORY_CACHE, None]:

            for shuffling_queue_capacity in [0, 20]:
                print("testing args: cache_type: {}, shuffling_queue_capacity: {}"
                      .format(cache_type,
                              shuffling_queue_capacity))

                if cache_type == MEMORY_CACHE:
                    make_fn_params = dict(cache_size_limit=1000,
                                          cache_row_size_estimate=1,
                                          cache_in_loader_memory=True,
                                          num_epochs=1)
                else:
                    make_fn_params = \
                        dict(shuffling_queue_capacity=shuffling_queue_capacity,
                             num_epochs=2)
                print("make_fn_params", make_fn_params)

                with make_batched_reader_and_loader(
                    batch_size=batch_size,
                    transform_fn=partial(torch.as_tensor, device='cpu'),
                    dataset_url_or_urls=self._dataset_url,
                    cur_shard=0,
                    shard_count=1,
                    reader_pool_type='thread',
                    workers_count=2,
                    hdfs_driver='libhdfs',
                    schema_fields=['col_0'],
                    **make_fn_params) as loader:

                    it = iter(loader)
                    retrieved_so_far = None

                    for _ in range(5):
                        batch = next(it)
                        this_batch = batch['col_0'].clone()
                        assert list(this_batch.shape)[0] == batch_size

                        if retrieved_so_far is None:
                            retrieved_so_far = this_batch
                        else:
                            if cache_type == MEMORY_CACHE:
                                intersect = set(retrieved_so_far.tolist()). \
                                    intersection(set(this_batch.tolist()))
                                assert not intersect
                            retrieved_so_far = torch.cat([retrieved_so_far, this_batch], 0)

                    if cache_type == MEMORY_CACHE:
                        assert len(set(retrieved_so_far.tolist())) == self._num_rows

                    for _ in range(5):
                        batch = next(it)
                        if cache_type == MEMORY_CACHE:
                            assert loader._shuffling_buffer._done_adding
                        this_batch = batch['col_0'].clone()
                        assert list(this_batch.shape)[0] == batch_size
                        retrieved_so_far = torch.cat([retrieved_so_far, this_batch], 0)

                    with pytest.raises(StopIteration):
                        next(it)

    def test_in_memory_cache_one_epoch(self):
        batch_size = 10
        for reader_factory in [make_batch_reader, make_reader]:
            for cache_type in [None, MEMORY_CACHE]:
                for shuffling_queue_capacity in [20, 0]:
                    print("testing reader_factor: {}, cache_type: {}, shuffling_queue_capacity: {}"
                          .format(reader_factory.__name__,
                                  cache_type,
                                  shuffling_queue_capacity))

                    if cache_type == 'mem_cache':
                        extra_loader_params = dict(cache_size_limit=1000,
                                                   cache_row_size_estimate=1,
                                                   cache_in_loader_memory=True,
                                                   num_epochs_to_load=1)
                        extra_reader_params = dict(num_epochs=1)
                        print("extra_reader_params", extra_reader_params)
                    else:
                        extra_loader_params = \
                            dict(shuffling_queue_capacity=shuffling_queue_capacity)
                        extra_reader_params = dict(num_epochs=1)

                    with reader_factory(self._dataset_url,
                                        cur_shard=0,
                                        shard_count=1,
                                        reader_pool_type='thread',
                                        workers_count=2,
                                        hdfs_driver='libhdfs',
                                        schema_fields=['col_0'],
                                        **extra_reader_params) as reader:

                        loader = BatchedDataLoader(reader,
                                                   batch_size=batch_size,
                                                   transform_fn=partial(torch.as_tensor,
                                                                        device='cpu'),
                                                   **extra_loader_params)

                        it = iter(loader)
                        retrieved_so_far = None

                        for _ in range(5):
                            batch = next(it)
                            this_batch = batch['col_0'].clone()

                            assert list(this_batch.shape)[0] == batch_size

                            if retrieved_so_far is None:
                                retrieved_so_far = this_batch
                            else:
                                intersect = set(retrieved_so_far.tolist()). \
                                    intersection(set(this_batch.tolist()))
                                assert not intersect
                                retrieved_so_far = torch.cat([retrieved_so_far, this_batch], 0)

                        assert len(set(retrieved_so_far.tolist())) == self._num_rows

                        with pytest.raises(StopIteration):
                            next(it)
