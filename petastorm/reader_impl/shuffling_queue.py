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
class ShufflingQueueBase(object):
    @abc.abstractmethod
    def enqueue_many(self, data):
        pass

    @abc.abstractmethod
    def dequeue(self):
        pass

    @abc.abstractmethod
    def can_enqueue(self):
        pass

    @abc.abstractmethod
    def can_dequeue(self):
        pass

    @abc.abstractmethod
    def size(self):
        return self._size


class NoopQueue(ShufflingQueueBase):
    def __init__(self):
        self.store = []

    def enqueue_many(self, data):
        self.store.extend(data)

    def dequeue(self):
        return self.store.pop(0)

    def can_dequeue(self):
        return len(self.store) > 0

    def can_enqueue(self):
        return True

    def size(self):
        return len(self.store)


class ShufflingQueue(ShufflingQueueBase):
    _EXTRA_QUEUE_CAPACITY = 1000

    def __init__(self, shuffling_queue_capacity, min_after_dequeue):
        self._data_list = [None] * (shuffling_queue_capacity + ShufflingQueue._EXTRA_QUEUE_CAPACITY)
        self._shuffling_queue_capacity = shuffling_queue_capacity
        self._min_after_dequeue = min_after_dequeue
        self._size = 0

    def enqueue_many(self, data):
        if not self.can_enqueue():
            raise RuntimeError('Can not enqueue. Check the return value of "can_enqueue()" to check if more '
                               'items can be added.')

        expected_size = self.size + len(data)
        maximal_capacity = self._shuffling_queue_capacity + ShufflingQueue._EXTRA_QUEUE_CAPACITY
        if expected_size > maximal_capacity:
            raise RuntimeError('Attempt to enqueue more elements that the capacity allows. '
                               'Current size: {}, new size {}, maximum allowed: {}'.format(self.size, expected_size,
                                                                                           maximal_capacity))
        self._data_list[self._size:len(data)] = data
        self._size = expected_size

    def dequeue(self):
        if not self.can_dequeue():
            raise RuntimeError('Can not dequeue. Check the return value of "can_dequeue()" to check if any '
                               'items are available.')
        random_index = np.random.randint(0, self._size)
        return_value = self._data_list[random_index]
        self._data_list[random_index] = self._data_list[self._size - 1]
        self._size -= 1
        return return_value

    def can_enqueue(self):
        return self._size < self._shuffling_queue_capacity

    def can_dequeue(self):
        return self._size >= self._min_after_dequeue

    @property
    def size(self):
        return self._size
