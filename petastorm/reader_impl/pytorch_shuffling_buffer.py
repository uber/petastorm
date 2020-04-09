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

import abc
from collections import deque

import six
import torch


@six.add_metaclass(abc.ABCMeta)
class BatchedShufflingBufferBase(object):
    """Shuffling implements a shuffling algorithm. Items can be added to the shuffling buffer and removed in a
    different order as defined by the concrete shuffling algorithm. A shuffling buffer is intended to be used from
    a single thread, hence, not thread safe.
    Functionality is similar to ShufflingBufferBase except operations are batched and base on PyTorch."""

    def __init__(self, batch_size=1):
        self._keys = None
        self.batch_size = batch_size

    def add_many(self, items):
        if isinstance(items, dict):
            if self._keys is None:
                self._keys = list(items.keys())
            items = [items[k] for k in self._keys]
        items = [torch.as_tensor(i) for i in items]
        if items[0].shape == ():
            # single value buffer
            self._keys = ""
            items = [torch.stack(items, 0)]

        return self._add_many(items)

    @abc.abstractmethod
    def _add_many(self, items):
        """Adds multiple items to the buffer.

        :param items: items to be added to the shuffling buffer.
        :return: None
        """

    def retrieve(self):
        sample = self._retrieve()
        if self._keys is not None:
            if self._keys == "":
                return sample[0].item()
            return {k: v for k, v in zip(self._keys, sample)}
        return sample

    @abc.abstractmethod
    def _retrieve(self):
        """Selects an item from the buffer and returns the item to the caller. The item is removed from the buffer.

        :return: The selected item.
        """

    @abc.abstractmethod
    def can_add(self):
        """Checks the state of the buffer and returns whether a new item can be added to the buffer at the time.

        :return: A boolean indicating whether an item can be added to the buffer at the time.
        """

    @abc.abstractmethod
    def can_retrieve(self):
        """Checks the state of the buffer and returns whether an item can be removed from the buffer..

        :return: A boolean indicating whether an item can be returned from the buffer at the time.
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
        super().__init__(batch_size=batch_size)
        self._batches = []
        self._num_samples = 0
        self.store = deque()

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
        self._batches.append(items)
        while self._num_samples >= self.batch_size:
            self._make_batch()

    def _retrieve(self):
        return self.store.popleft()

    def can_retrieve(self):
        return len(self.store) > 0

    def can_add(self):
        return True

    @property
    def size(self):
        return len(self.store)

    def finish(self):
        if self._batches:
            self._make_batch()


class BatchedRandomShufflingBuffer(BatchedShufflingBufferBase):
    """
    A random shuffling buffer implementation. Items can be added to the buffer and retrieved in a random order.
    """

    def __init__(self, shuffling_buffer_capacity, min_after_retrieve, extra_capacity=1000, batch_size=1):
        """Initializes a new ShufflingBuffer instance.

        Items may be retrieved from the buffer once ``min_after_retrieve`` items were added to the queue
        (indicated by ``can_retrieve``).

        Items may be added to the buffer as long as the number of items in the buffer (not including the items
        passed to :func:`add_many`) does not exceed ``shuffling_queue_capacity``.

        The amount of items in the buffer may actually become more than ``shuffling_buffer_capacity`` since
        :func:`add_many` is passed a list of items. The *hard limit* on the number of items in the buffer is
        ``shuffling_buffer_capacity + extra_capacity``.

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
        """
        super().__init__(batch_size=batch_size)
        self._extra_capacity = extra_capacity
        # Preallocate the shuffling buffer.
        self._items = None
        self._keys = None
        self._shuffling_queue_capacity = shuffling_buffer_capacity
        self._min_after_dequeue = min_after_retrieve
        self._size = 0
        self._done_adding = False

        self._random_indices = None
        self._sampling_size = 0

    def _add_many(self, items):
        if self._done_adding:
            raise RuntimeError('Can not call add_many after done_adding() was called.')

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

        if self._sampling_size > 0:
            # Before we can append a new batch, we should remove used samples
            for k, v in enumerate(self._items):
                # We need to clone the right-side to avoid racing conditions
                self._items[k][:self.size] = self._items[k][self._random_indices[self._sampling_size:]].clone()
        self._random_indices = None
        self._sampling_size = 0

        if new_capacity > self._items[0].shape[0]:
            for k, v in enumerate(self._items):
                self._items[k] = torch.empty((new_capacity,) + v.shape[1:], dtype=v.dtype, device=v.device)
                self._items[k][:self._size] = v[:self._size]

        # Copy new items over
        for k, v in enumerate(items):
            self._items[k][self._size:expected_size] = v
        self._size = expected_size

    def _retrieve(self):
        if not self._done_adding and not self.can_retrieve():
            raise RuntimeError('Can not dequeue. Check the return value of "can_dequeue()" to check if any '
                               'items are available.')
        batch_size = min(self.batch_size, self._size)

        if self._random_indices is None:
            self._sampling_size = 0
            self._random_indices = torch.randperm(int(self._size), device=self._items[0].device)
        idx = self._random_indices[self._sampling_size:self._sampling_size + batch_size]
        self._sampling_size += batch_size
        sample = []
        for v in self._items:
            # Clone is required because pytorch doesn't always make a copy
            sample.append(v[idx])
        self._size -= batch_size
        return sample

    def can_add(self):
        return self._size <= self._shuffling_queue_capacity - self.batch_size and not self._done_adding

    def can_retrieve(self):
        return self._size >= self._min_after_dequeue + self.batch_size - 1 or (self._done_adding and self._size > 0)

    @property
    def size(self):
        return self._size

    def finish(self):
        self._done_adding = True
