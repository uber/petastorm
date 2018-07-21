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

import numbers


class Sequence(object):
    """
    Defines a sequence, of length items, where consecutive items in a sequence are no further apart than
    delta_threshold (inclusive) (where the field that indicate the timestamp of the item being timestamp_field).

    The length parameter defines how many elements will be in each sequence.

    The delta_threshold and timestamp_field defines how far away each item in the sequence will be as described in the
    rules above.

    The following are examples of what sequences will return based on the parameters:

    Case 1: length = 2, delta_threshold=5, timestamp_field='ts'

            The data being:
                A {'timestamp': 0,  ....}
                B {'timestamp': 10, ....}
                C {'timestamp': 20, ....}
                D {'timestamp': 30, ....}

            The result will be empty, since delta is 10 (more than allowed delta_threshold of 5)

    Case 2: length = 2, delta_threshold=4, timestamp_field='ts'

            The data being:
                A {'timestamp': 0, .....}
                B {'timestamp': 3, .....}
                C {'timestamp': 8, .....}
                D {'timestamp': 10, .....}
                E {'timestamp': 11, .....}
                G {'timestamp': 20, .....}
                H {'timestamp': 30, .....}

            The result will be (A, B), (C, D), (D, E)
            Notice how:
            - (B, C) was skipped since B timestamp is 3 and C timestamp is 8 (difference of 5 >= delta_threshold is 4)
            - (E, G), (G, H) were also skipped due to the same reason

    One caveat to note, that all sequences within the same parquet piece are guaranteed to be returned, but not across
    different parquet pieces. i.e. if piece 1 has [0, 5], piece 2 has [6, 10] then this will result in
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9), (9, 10)
    Notice how the (5, 6) was skipped because it is across two different pieces

    The return type will be a dictionary where the keys are [0, length[ and the value of each key will be the item.
    """

    def __init__(self, length, delta_threshold, timestamp_field):
        """
        Constructor to initialize sequence with length, delta_threshold and timestamp_field.
        :param length: The sequence length.
        :param delta_threshold: The maximum threshold of delta between timestamp_field
        :param timestamp_field: The field that represents the timestamp
        """
        self._length = length
        self._delta_threshold = delta_threshold
        self._timestamp_field = timestamp_field

        self._validate_sequence(length, delta_threshold, timestamp_field)

    @property
    def length(self):
        """
        return the sequence length requested.
        :return: the sequence length.
        """
        return self._length

    @property
    def delta_threshold(self):
        """
        The maximum difference between one entry and the following one in timestamp_field.
        :return: The delta threshold.
        """
        return self._delta_threshold

    @property
    def timestamp_field(self):
        """
        The field of the entry that represent the timestamp.
        :return: timestamp field.
        """
        return self._timestamp_field

    def _validate_sequence(self, length, delta_threshold, timestamp_field):
        """
        Validates the length, delta_threshold and timestamp_field are set and of the correct types.
        :param length: The sequence length.
        :param delta_threshold: The delta threshold.
        :param timestamp_field: The timestamp field.
        """
        if length is None or not isinstance(length, numbers.Integral):
            raise ValueError('length must be set and must be an integer.')

        if length is None or not isinstance(delta_threshold, numbers.Number):
            raise ValueError('delta_threshold must be a number.')

        if timestamp_field is None or not isinstance(timestamp_field, str):
            raise ValueError('if timestamp_field must be set and must be a string.')

    def _sequence_pass_threshold(self, sequence):
        """
        Returns true if each item in a sequence passes the threshold, otherwise False.
        Specifically that means that timestamp of an item - timestamp of previous item <= delta_threshold

        :param sequence: An array of items
        :return: True if each item in a sequence passes threshold, otherwise False.
        """
        # Verify that each element and it's previous element do not exceed the delta_threshold
        for previous, current in zip(sequence[:-1], sequence[1:]):
            if getattr(current, self.timestamp_field) - getattr(previous, self.timestamp_field) > self.delta_threshold:
                return False
        return True

    def form_sequence(self, data):
        """
        Return all the sequences as dictated by length, delta_threshold and timestamp_field.
        :param data: The data items.
        :return: A dictionary, with keys [0, length - 1]. The value of each key is the corresponding item in the
        sequence at that position.
        """

        result = []

        for index in range(len(data) - self.length + 1):
            # Potential sequence: [index, index + self.length[
            potential_sequence = data[index:index + self.length]

            # If all elements in potential_sequence passes the sequence threshold
            # (i.e. current element timestamp - previous element timestamp <= delta_threshold)
            # then add the potential sequence in the results, otherwise skip it
            if self._sequence_pass_threshold(potential_sequence):
                result.append({key: value for (key, value) in enumerate(potential_sequence)})

        return result
