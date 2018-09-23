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

import numpy as np
import six


@six.add_metaclass(abc.ABCMeta)
class ShufflingBufferBase(object):
    """Shuffling implements a shuffling algorithm. Items can be added to the shuffling buffer and removed in a
    different order as defined by the concrete shuffling algorithm. A shuffling buffer is intended to be used from
    a single thread, hence, not thread safe."""
    @abc.abstractmethod
    def add_many(self, items):
        """Adds multiple items to the buffer.

        :param items: items to be added to the shuffling buffer.
        :return: None
        """
        pass

    @abc.abstractmethod
    def retrieve(self):
        """Selects an item from the buffer and returns the item to the caller. The item is removed from the buffer.

        :return: The selected item.
        """
        pass

    @abc.abstractmethod
    def can_add(self):
        """Checks the state of the buffer and returns whether a new item can be added to the buffer at the time.

        :return: A boolean indicating whether an item can be added to the buffer at the time.
        """
        pass

    @abc.abstractmethod
    def can_retrieve(self):
        """Checks the state of the buffer and returns whether an item can be removed from the buffer..

        :return: A boolean indicating whether an item can be returned from the buffer at the time.
        """
        pass

    @abc.abstractproperty
    def size(self):
        """Returns the number of elements currently present in the buffer.

        :return: number of elements currently present in the buffer
        """
        pass

    @abc.abstractmethod
    def finish(self):
        """Call this method when no more :func:`add_many` calls will be made.

        This allows a user to deplete the buffer. Typically during last epoch. Otherwise, we would always have leftovers
        in the buffer at the end of the lifecycle.

        :return: number of elements currently present in the buffer
        """
        pass


class NoopShufflingBuffer(ShufflingBufferBase):
    """A 'no-operation' (noop) implementation of a shuffling buffer. Useful in cases where no shuffling is desired, such
    as test scenarios or iterating over a dataset in a predeterministic order.
    """

    def __init__(self):
        self.store = []

    def add_many(self, items):
        self.store.extend(items)

    def retrieve(self):
        return self.store.pop(0)

    def can_retrieve(self):
        return len(self.store) > 0

    def can_add(self):
        return True

    @property
    def size(self):
        return len(self.store)

    def finish(self):
        pass


class RandomShufflingBuffer(ShufflingBufferBase):
    """
    A random shuffling buffer implementation. Items can be added to the buffer and retrieved in a random order.
    """

    def __init__(self, shuffling_buffer_capacity, min_after_retrieve, extra_capacity=1000):
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
        self._extra_capacity = extra_capacity
        # Preallocate the shuffling buffer.
        self._items = [None] * (shuffling_buffer_capacity + self._extra_capacity)
        self._shuffling_queue_capacity = shuffling_buffer_capacity
        self._min_after_dequeue = min_after_retrieve
        self._size = 0
        self._done_adding = False

    def add_many(self, items):
        if self._done_adding:
            raise RuntimeError('Can not call add_many after done_adding() was called.')

        if not self.can_add():
            raise RuntimeError('Can not enqueue. Check the return value of "can_enqueue()" to check if more '
                               'items can be added.')

        # We leave self._extra_capacity slack to make sure we don't reallocate self._items array
        expected_size = self._size + len(items)
        maximal_capacity = self._shuffling_queue_capacity + self._extra_capacity
        if expected_size > maximal_capacity:
            raise RuntimeError('Attempt to enqueue more elements that the capacity allows. '
                               'Current size: {}, new size {}, maximum allowed: {}'.format(self._size, expected_size,
                                                                                           maximal_capacity))
        self._items[self._size:self._size + len(items)] = items
        self._size = expected_size

    def retrieve(self):
        if not self._done_adding and not self.can_retrieve():
            raise RuntimeError('Can not dequeue. Check the return value of "can_dequeue()" to check if any '
                               'items are available.')
        random_index = np.random.randint(0, self._size)
        return_value = self._items[random_index]
        self._items[random_index] = self._items[self._size - 1]
        self._items[self._size - 1] = None
        self._size -= 1
        return return_value

    def can_add(self):
        return self._size < self._shuffling_queue_capacity and not self._done_adding

    def can_retrieve(self):
        return self._size >= self._min_after_dequeue or (self._done_adding and self._size > 0)

    @property
    def size(self):
        return self._size

    def finish(self):
        self._done_adding = True
