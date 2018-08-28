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

import random


def epoch_generator(items, num_epochs, shuffle):
    """This generator is used to generate epochs.

    An epoch is a single pass on a set of data (without repetitions). If shuffle is False, items are yielded in
    the same order as passed into the generator. If shuffle is True, the items order is reshuffled for each epoch.
    If num_epochs is a None, infinite number of epochs will be generated. 'items' may contain any type of an
    object/primitive.

    Example:
      assert list(epoch_generator([1, 2, 3], 1, False)) == [1, 2, 3]
      assert list(epoch_generator([1, 2, 3], 2, False)) == [1, 2, 3, 1, 2, 3]

    :param items: List of objects in a single epoch.
    :param num_epochs: Number of epochs to generate
    :param shuffle: If True, the order of items in each epoch is randomized
    :return:
    """

    if num_epochs is not None and (not isinstance(num_epochs, int) or num_epochs < 1):
        raise ValueError('iterations must be positive integer or None')

    curr_item_index = 0

    epochs_left = num_epochs

    # Continue until 'epochs_left' is greater than 0. If 'None', continue forever
    while (epochs_left is None or epochs_left > 0) and items:

        if curr_item_index == 0 and shuffle:
            random.shuffle(items)

        yield items[curr_item_index]

        curr_item_index += 1

        if curr_item_index >= len(items):
            curr_item_index = 0

            # If iterations was set to None, that means we will iterate until stop is called
            if epochs_left is not None:
                epochs_left -= 1
