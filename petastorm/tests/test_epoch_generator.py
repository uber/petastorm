import pytest

from petastorm.reader_impl.epochs import epoch_generator


def test_one_epoch():
    assert list(epoch_generator([1, 2, 3], 1, False)) == [1, 2, 3]


def test_three_epochs():
    assert list(epoch_generator([1, 2, 3], 3, False)) == [1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_shuffle():
    items_in_epoch = 50
    shuffled_epoch = list(epoch_generator(list(range(items_in_epoch)), 1, True))
    assert len(shuffled_epoch) == items_in_epoch
    assert shuffled_epoch != range(items_in_epoch)


def test_shuffle_per_epoch():
    items_in_epoch = 50
    two_epochs = list(epoch_generator(list(range(items_in_epoch)), 2, True))
    assert len(two_epochs) == 100
    assert two_epochs[:50] != two_epochs[50:]


def test_empty_items_list():
    assert not list(epoch_generator([], 2, True))
    assert not list(epoch_generator([], None, True))


def test_item_type_can_be_anything():
    assert list(epoch_generator([None, 'a string', {}], 1, False)) == [None, 'a string', {}]


def test_unlimited_epochs():
    items = [1, 2, 3]
    inf_epochs = epoch_generator(items, None, True)
    for _ in range(1000):
        actual_item = next(inf_epochs)
        assert actual_item in items


def test_one_item_epoch():
    assert list(epoch_generator([1, ], 3, False)) == [1, 1, 1]


def test_invalid_number_of_epochs():
    with pytest.raises(ValueError):
        list(epoch_generator([0, 1], 0, False))

    with pytest.raises(ValueError):
        list(epoch_generator([0, 1], -1, False))

    with pytest.raises(ValueError):
        list(epoch_generator([0, 1], 'abc', False))
