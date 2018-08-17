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

import numpy as np
import pytest
import six

from petastorm.reader_impl.shuffling_buffer import NoopShufflingBuffer, RandomShufflingBuffer


def test_noop_shuffling_buffer():
    """Noop should not do any shuffling. Add/retrieve some items while checking correcness of can_retrieve"""
    q = NoopShufflingBuffer()

    # Empty buffer. Can add, can not retrieve with zero size
    assert q.size == 0
    assert q.can_add()
    assert not q.can_retrieve()

    # Try adding some items. Check queue size and can_retrieve indicator
    q.add_many([1])
    assert q.can_add()
    assert q.size == 1

    q.add_many([2, 3])
    assert q.size == 3

    assert 1 == q.retrieve()
    assert q.can_retrieve()

    assert 2 == q.retrieve()
    assert 3 == q.retrieve()
    assert not q.can_retrieve()

    # No effect is expected in noop implementation
    q.finish()


def test_random_shuffling_buffer_can_add_retrieve_flags():
    """Check can_add/can_retrieve flags at all possible states"""
    q = RandomShufflingBuffer(5, 3)

    # Empty buffer. Can start adding, nothing to retrieve yet
    assert q.size == 0
    assert q.can_add()
    assert not q.can_retrieve()

    # Under min_after_retrieve elements, so can not retrieve just yet
    q.add_many([1, 2])
    assert q.can_add()
    assert not q.can_retrieve()
    assert q.size == 2

    # Got to min_after_retrieve elements, can start retrieving
    q.add_many([3])
    assert q.can_retrieve()
    assert q.size == 3

    # But when we retrieve we are again under min_after_retrieve, so can not retrieve again
    q.retrieve()
    assert not q.can_retrieve()
    assert q.size == 2

    # Getting back to the retrievable state with enough items in the buffer
    q.add_many([4, 5])
    assert q.can_add()
    assert q.can_retrieve()
    assert q.size == 4

    # Can overrun the capacity (as long as below extra_capacity), but can not add if we are above
    # shuffling_buffer_capacity
    q.add_many([6, 7, 8, 9])
    assert not q.can_add()
    with pytest.raises(RuntimeError):
        q.add_many([1])
    assert q.can_retrieve()
    assert q.size == 8

    # Getting one out. Still have more than shuffling_buffer_capacity
    q.retrieve()
    assert not q.can_add()
    assert q.can_retrieve()
    assert q.size == 7

    # Retrieve enough to get back to addable state
    [q.retrieve() for _ in range(4)]
    assert q.can_add()
    assert q.can_retrieve()
    assert q.size == 3

    # Retrieve the last element so we go under min_after_retrieve and can not retrieve any more
    q.retrieve()
    assert q.can_add()
    assert not q.can_retrieve()
    with pytest.raises(RuntimeError):
        q.retrieve()

    assert q.size == 2

    # finish() will allow us to deplete the buffer completely
    q.finish()
    assert not q.can_add()
    assert q.can_retrieve()
    assert q.size == 2

    q.retrieve()
    assert not q.can_add()
    assert q.can_retrieve()
    assert q.size == 1

    q.retrieve()
    assert not q.can_add()
    assert not q.can_retrieve()
    assert q.size == 0


def _feed_a_sequence_through_the_queue(shuffling_buffer, input_sequence):
    assert shuffling_buffer.size == 0
    assert not shuffling_buffer.can_retrieve()

    retrieve_sequence = []
    fro = 0
    while True:
        if shuffling_buffer.can_add():
            to = min(fro + 3, len(input_sequence))
            next_input_chunk = input_sequence[fro:to]
            fro = to

            shuffling_buffer.add_many(next_input_chunk)

            if to >= len(input_sequence):
                shuffling_buffer.finish()
                break

        for _ in range(2):
            if shuffling_buffer.can_retrieve():
                retrieve_sequence.append(shuffling_buffer.retrieve())

    while shuffling_buffer.can_retrieve():
        retrieve_sequence.append(shuffling_buffer.retrieve())

    return retrieve_sequence


def test_random_shuffling_buffer_stream_through():
    """Feed a 0:99 sequence through a RandomShufflingBuffer. Check that the order has changed."""
    input_sequence = range(100)
    a = _feed_a_sequence_through_the_queue(RandomShufflingBuffer(10, 3), input_sequence)
    b = _feed_a_sequence_through_the_queue(RandomShufflingBuffer(10, 3), input_sequence)
    assert len(a) == len(input_sequence)
    assert set(a) == set(b)
    assert a != b


def test_noop_shuffling_buffer_stream_through():
    """Feed a 0:99 sequence through a NoopShufflingBuffer. Check that the order has not changed."""
    expected = list(range(100))
    actual = _feed_a_sequence_through_the_queue(NoopShufflingBuffer(), expected)
    assert expected == actual


def test_longer_random_sequence_of_queue_ops():
    """A long random sequence of added and retrieved values"""
    q = RandomShufflingBuffer(100, 80)

    for _ in six.moves.xrange(10000):
        if q.can_add():
            q.add_many(np.random.random((np.random.randint(1, 10),)))
        assert q.size < 100 + 10
        for _ in range(np.random.randint(1, 10)):
            if not q.can_retrieve():
                break
            # Make sure never get to less than `min_after_retrieve` elements
            assert 80 <= q.size
            q.retrieve()
