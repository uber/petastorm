import numpy as np
import pytest
import six

from petastorm.reader_impl.shuffling_queue import NoopQueue, ShufflingQueue


def test_noop_queue():
    q = NoopQueue()
    assert q.size() == 0
    assert not q.can_dequeue()

    q.enqueue_many([1])
    assert q.size() == 1

    q.enqueue_many([2, 3])
    assert q.size() == 3

    assert 1 == q.dequeue()
    assert q.can_dequeue()

    assert 2 == q.dequeue()
    assert 3 == q.dequeue()
    assert not q.can_dequeue()


def test_shuffling_queue():
    def feed_a_sequence_through_the_queue():
        q = ShufflingQueue(10, 3)
        assert q.size == 0
        assert not q.can_dequeue()

        q.enqueue_many(range(3))
        enqueue_value = 3

        dequeue_sequence = []
        for _ in range(50):
            q.enqueue_many([enqueue_value, enqueue_value + 1])
            enqueue_value += 2

            dequeue_sequence.append(q.dequeue())
            dequeue_sequence.append(q.dequeue())

        return dequeue_sequence

    a = feed_a_sequence_through_the_queue()
    b = feed_a_sequence_through_the_queue()
    assert a != b


def test_deque_when_not_enough_items_should_raise():
    q = ShufflingQueue(10, 3)
    assert q.size == 0
    assert not q.can_dequeue()

    with pytest.raises(RuntimeError):
        q.dequeue()

    q.enqueue_many([1])
    with pytest.raises(RuntimeError):
        q.dequeue()

    q.enqueue_many([2])
    with pytest.raises(RuntimeError):
        q.dequeue()

    q.enqueue_many([3])
    q.dequeue()


def test_enqueue_above_capacity_shoud_raise():
    q = ShufflingQueue(10, 3)
    q.enqueue_many(range(10))
    with pytest.raises(RuntimeError):
        q.enqueue_many(1)


def test_longer_random_sequence_of_queue_ops():
    q = ShufflingQueue(100, 80)

    for _ in six.moves.xrange(10000):
        if q.can_enqueue():
            q.enqueue_many(np.random.random((np.random.randint(1, 10),)))
        assert q.size < 100 + 10
        for _ in range(np.random.randint(1, 10)):
            if not q.can_dequeue():
                break
            assert 80 <= q.size
            q.dequeue()
