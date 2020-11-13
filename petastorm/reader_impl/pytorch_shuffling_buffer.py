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

import abc
from collections import deque

import six
import torch


@six.add_metaclass(abc.ABCMeta)
class BatchedShufflingBufferBase(object):
    """Shuffling implements a shuffling algorithm. Items can be added to the shuffling buffer and removed in a
    different order as defined by the concrete shuffling algorithm. A shuffling buffer is intended to be used from
    a single thread, hence, not thread safe.
    Functionality is similar to ShufflingBufferBase except operations are batched and based on PyTorch."""

    def __init__(self, batch_size=1):
        self._keys = None
        self.batch_size = batch_size
        self._done_adding = False

    def add_many(self, items):
        items = [torch.as_tensor(v) for v in items]

        return self._add_many(items)

    @abc.abstractmethod
    def _add_many(self, items):
        """Adds multiple items to the buffer.

        :param items: items to be added to the shuffling buffer.
        :return: None
        """

    @abc.abstractmethod
    def retrieve(self):
        """Selects an batch of items from the buffer and returns the batch to the caller.
        The items are removed from the buffer.

        :return: The selected batch.
        """

    @abc.abstractmethod
    def can_add(self):
        """Checks the state of the buffer and returns whether a new item can be added to the buffer at the time.

        :return: A boolean indicating whether an item can be added to the buffer at the time.
        """

    @abc.abstractmethod
    def can_retrieve(self):
        """Checks the state of the buffer and returns whether a batch can be removed from the buffer..

        :return: A boolean indicating whether an batch can be returned from the buffer at the time.
        """

    @abc.abstractproperty
    def size(self):
        """Returns the number of elements currently present in the buffer.

        :return: number of elements currently present in the buffer
        """

    @abc.abstractmethod
    def finish(self):
        """Call this method when no more :func:`add_many` calls will be made.

        This allows a user to deplete the buffer. Typically during last epoch. Otherwise, we would always have leftovers
        in the buffer at the end of the lifecycle.

        :return: number of elements currently present in the buffer
        """


class BatchedNoopShufflingBuffer(BatchedShufflingBufferBase):
    """A 'no-operation' (noop) implementation of a shuffling buffer. Useful in cases where no shuffling is desired, such
    as test scenarios or iterating over a dataset in a predeterministic order.
    """

    def __init__(self, batch_size=1):
        super(BatchedNoopShufflingBuffer, self).__init__(batch_size=batch_size)
        self._batches = []
        self._num_samples = 0
        self.store = deque()
        self._size = 0

    def _make_batch(self):
        # TODO: Add test for the zip
        batch = [torch.cat(b, 0) for b in zip(*self._batches)]
        if self._num_samples > self.batch_size:
            leftover = [b[self.batch_size:] for b in batch]
            batch = [b[:self.batch_size] for b in batch]
            self._batches = [leftover]
        else:
            self._batches = []
        self._num_samples -= min(self._num_samples, self.batch_size)
        self.store.append(batch)

    def _add_many(self, items):
        self._num_samples += len(items[0])
        self._size += len(items[0])
        self._batches.append(items)
        while self._num_samples >= self.batch_size:
            self._make_batch()

    def retrieve(self):
        batch = self.store.popleft()
        self._size -= len(batch[0])
        return batch

    def can_retrieve(self):
        return len(self.store) > 0

    def can_add(self):
        return True

    @property
    def size(self):
        return self._size

    def finish(self):
        if self._batches:
            self._make_batch()


class BatchedRandomShufflingBuffer(BatchedShufflingBufferBase):
    """
    A random shuffling buffer implementation. Items can be added to the buffer and retrieved in a random order.
    """

    def __init__(self, shuffling_buffer_capacity, min_after_retrieve, extra_capacity=1000, batch_size=1):
        """Initializes a new BatchedRandomShufflingBuffer instance.

        Items may be retrieved from the buffer once ``min_after_retrieve`` items were added to the queue
        (indicated by ``can_retrieve``).

        Items may be added to the buffer as long as the number of items in the buffer (not including the items
        passed to :func:`add_many`) does not exceed ``shuffling_queue_capacity``.

        The amount of items in the buffer may actually become more than ``shuffling_buffer_capacity`` since
        :func:`add_many` is passed a list of items. The *hard limit* on the number of items in the buffer is
        ``shuffling_buffer_capacity + extra_capacity``.

        Explanation:
        This batch loader performs some non-conventional operations:

        Let's say we enqueued several samples:

        [1, 2, 3, 4, 5, 6, 7]

        Now during a retrieve() we sample the order these samples will be retrieved:

        [2, 4, 5, 1, 3, 0, 6]

        Once an order has been sampled, we slice the order into batches of ``batch_size`` samples.
        And index 1 batch at a time:

        [1, 2, X, 4, X, 6, 7] -> [3, 5] (batch 1)
        [1, X, X, 4, X, X, 7] -> [6 ,2] (batch 2)

        We could compress the buffer after every retrieve(), but that would require custom ops.

        When we call add_many we first rearrange the remaining elements:

        [1, 4, 7]

        Then append new elements:
        [1, 4, 7, 8, 9, 10]

        After add_many we have to resample a permutation for the buffer.

        :param shuffling_buffer_capacity: Items may be added to the buffer as long as the amount of items in the
          buffer does not exceed the value of ``shuffling_queue_capacity`` (not including the items
          passed to :func:`add_many`).
        :param min_after_retrieve: Minimal amount of items in the buffer that allows retrieval. This is needed to
          guarantee good random shuffling of elements. Once :func:`finish` is called, items can be retrieved even if
          the condition does not hold.
        :param extra_capacity: The amount of items in the buffer may grow above ``shuffling_buffer_capacity``
          (due to a call to :func:`add_many` with a list of items), but must remain under ``extra_capacity``. Should be
          set to the upper bound of the number of items that can be added in a single call to :func:`add_many` (can be a
          loose bound).
        :param batch_size: The number of items to be retrieved for each self.retrieve() call.
        This also affects the can_add and can can_retrieve accordingly.
        """
        super(BatchedRandomShufflingBuffer, self).__init__(batch_size=batch_size)
        self._extra_capacity = extra_capacity
        # Preallocate the shuffling buffer.
        self._items = None
        self._shuffling_queue_capacity = shuffling_buffer_capacity
        self._min_after_dequeue = min_after_retrieve
        self._size = 0
        self._done_adding = False

        self._random_indices = None
        self.next_sample_head = 0

    def _add_many(self, items):
        if self._done_adding:
            raise RuntimeError('Can not call add_many after done_adding() was called.')

        if not self.can_add():
            raise RuntimeError('Can not enqueue. Check the return value of "can_enqueue()" to check if more '
                               'items can be added.')

        expected_size = self._size + len(items[0])
        maximal_capacity = self._shuffling_queue_capacity + self._extra_capacity
        if expected_size > maximal_capacity:
            raise RuntimeError('Attempt to enqueue more elements than the capacity allows. '
                               'Current size: {}, new size {}, maximum allowed: {}'.format(self._size, expected_size,
                                                                                           maximal_capacity))

        new_capacity = self._shuffling_queue_capacity
        while new_capacity < expected_size:
            # Will double capacity until it is large enough to fit new batch
            new_capacity *= 2

        if self._items is None:
            # Create Buffer:
            self._items = []
            for v in items:
                self._items.append(torch.empty((new_capacity,) + v.shape[1:], dtype=v.dtype, device=v.device))

        if self.next_sample_head > 0:
            # Before we can append a new batch, we compress the remaining samples
            for k, v in enumerate(self._items):
                # We need to clone the right-side to avoid racing conditions
                self._items[k][:self.size] = self._items[k][self._random_indices[self.next_sample_head:]].clone()
        self._random_indices = None
        self.next_sample_head = 0

        if new_capacity > self._items[0].shape[0]:
            for k, v in enumerate(self._items):
                self._items[k] = torch.empty((new_capacity,) + v.shape[1:], dtype=v.dtype, device=v.device)
                self._items[k][:self._size] = v[:self._size]

        # Copy new items over
        for k, v in enumerate(items):
            self._items[k][self._size:expected_size] = v
        self._size = expected_size

    def retrieve(self):
        if not self._done_adding and not self.can_retrieve():
            raise RuntimeError('Can not dequeue. Check the return value of "can_dequeue()" to check if any '
                               'items are available.')
        batch_size = min(self.batch_size, self._size)

        if self._random_indices is None:
            # We randomize the order of all samples ahead of time and then slice it into chunks with ```batch_size```
            self.next_sample_head = 0
            self._random_indices = torch.randperm(int(self._size), device=self._items[0].device)
        idx = self._random_indices[self.next_sample_head:self.next_sample_head + batch_size]
        self.next_sample_head += batch_size
        sample = [v[idx] for v in self._items]
        self._size -= batch_size
        return sample

    def can_add(self):
        return self._size < self._shuffling_queue_capacity and not self._done_adding

    def can_retrieve(self):
        return self._size >= self._min_after_dequeue + self.batch_size - 1 or (self._done_adding and self._size > 0)

    @property
    def size(self):
        return self._size

    def finish(self):
        self._done_adding = True


class BatchedRandomShufflingBufferWithMemCache(BatchedShufflingBufferBase):
    """
    A random shuffling buffer implementation that caches the added data in memory for fast loading
    and high performance.
    """

    def __init__(self, cache_size, batch_size=1, num_epochs_to_load=1):
        """Initializes a new BatchedRandomShufflingBufferWithMemCache instance.

        The buffer caches all the data in memory from reader and then start loading it.

        Items may be retrieved from the buffer once all the data is read. Data cannot be loaded
        before data reading is done.

        Items may be added to the buffer as long as the number of items in the buffer does not
        exceed the ``cache_size``. ``cache_size`` is a hard limit. This cache does not evict any
        members or invalidate existing items. That will defeat the purpose of this cache which
        is to provide a very high performance data access for the training and avoiding the disk or
        network bottleneck. The partition of data that is read has to fit in this cache,
        otherwise an error will be raised.

        Data will be shuffled once after reading is complete. There will be no automatic shuffling
        but user can trigger data shuffle by calling ``shuffle`` method.

        :param cache_size: The maximum number of items that can be stored in memory cache. We will
            throw an error if the number of elements grows beyond the cache_size.
        :param batch_size: The number of items to be retrieved for each self.retrieve() call.
            This also affects the can_add and can can_retrieve accordingly.
        :param num_epochs_to_load: Tells the buffer how many times every instance of the data
            will be loaded. When using a memory cache, the readers reads the data only once, but
            loader will load it num_epochs_to_load. num_epochs_to_load is essentially the number of
            training epochs from the model's point of view.
        """
        super(BatchedRandomShufflingBufferWithMemCache, self).__init__(batch_size=batch_size)
        self._num_epochs_to_load = num_epochs_to_load
        self._cache_size = cache_size

        # Preallocate the shuffling buffer.
        self._items = None
        self._size = 0

        self.next_sample_head = 0
        self._retrieved_samples_so_far = 0

    def shuffle(self):
        for k in range(len(self._items)):
            random_indices = torch.randperm(int(self._size), device=self._items[0].device)
            self._items[k][:self._size] = self._items[k][random_indices]

        self.next_sample_head = 0

    def _add_many(self, items):
        if self._done_adding:
            raise RuntimeError('Can not call add_many after done_adding() was called.')

        if not self.can_add():
            raise RuntimeError(
                'Can not enqueue. Check the return value of "can_enqueue()" to check if more '
                'items can be added.')

        expected_size = self._size + len(items[0])
        maximal_capacity = self._cache_size
        if expected_size > maximal_capacity:
            raise RuntimeError(
                'Attempt to cache more elements than the memory cache capacity allows. Please '
                'increase your cache size. Current size: {}, new size {}, '
                'maximum number of samples allowed: {}'.format(self._size, expected_size,
                                                               self._cache_size))

        # 2 is an arbitrary number to start to grow from there.
        new_capacity = 2
        while new_capacity < expected_size:
            # Will double capacity until it is large enough to fit new batch
            new_capacity *= 2

        if self._items is None:
            # Create Buffer:
            self._items = []
            for v in items:
                self._items.append(
                    torch.empty((new_capacity,) + v.shape[1:], dtype=v.dtype, device=v.device))

        if new_capacity > self._items[0].shape[0]:
            for k, v in enumerate(self._items):
                self._items[k] = torch.empty((new_capacity,) + v.shape[1:], dtype=v.dtype,
                                             device=v.device)
                self._items[k][:self._size] = v[:self._size]

        # Copy new items over
        for k, v in enumerate(items):
            self._items[k][self._size:expected_size] = v
        self._size = expected_size

    def retrieve(self):
        if not self.can_retrieve():
            raise RuntimeError(
                'Can not dequeue. Check the return value of "can_retrieve()" to check if any '
                'items are available.')

        batch_size = min(self.batch_size, self._size)
        if self.next_sample_head + batch_size < self._size:
            sample = [v[self.next_sample_head:self.next_sample_head + batch_size] for v in self._items]
        else:
            overflow_idx = self.next_sample_head + batch_size - self._size
            sample = [torch.cat([v[self.next_sample_head:self._size], v[:overflow_idx]], dim=0) for v in self._items]
        self.next_sample_head = (self.next_sample_head + batch_size) % self._size
        self._retrieved_samples_so_far += batch_size
        return sample

    def can_add(self):
        return self._size < self._cache_size

    def can_retrieve(self):
        if not self._done_adding:
            return False
        return self._num_epochs_to_load is None or \
            self._retrieved_samples_so_far < self._num_epochs_to_load * self._size

    @property
    def size(self):
        return self._size

    def finish(self):
        self._done_adding = True
        self.shuffle()
