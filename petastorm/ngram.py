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

from petastorm.unischema import UnischemaField, match_unischema_fields
from six import string_types


class NGram(object):
    """
    Defines an NGram, having certain fields as set by fields, where consecutive items in an NGram are no further apart
    than the argument delta_threshold (inclusive). The argument timestamp_field indicate which field refers to the
    timestamp in the data.

    The argument fields is a dictionary, where the keys are integers, and the value is an array of the Unischema fields
    to include at that timestep.

    The delta_threshold and timestamp_field defines how far away each item in the NGram will be as described in the
    rules above.

    The following are examples of what NGram will return based on the parameters:

    A. Case 1:

    >>> fields = {
    >>>  -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png,
    >>>       TestSchema.matrix],
    >>>   0: [TestSchema.id, TestSchema.id2,
    >>>       TestSchema.sensor_name],
    >>> }
    >>> delta_threshold = 5
    >>> timestamp_field = 'id'

      - The data being:

    >>> A {'id': 0,  ....}
    >>> B {'id': 10, ....}
    >>> C {'id': 20, ....}
    >>> D {'id': 30, ....}

      - The result will be empty, since delta is 10 (more than allowed delta_threshold of 5)

    B. Case 2:

    >>> fields = {
    >>>     -1: [TestSchema.id, TestSchema.id2, TestSchema.image_png,
    >>>          TestSchema.matrix],
    >>>      0: [TestSchema.id, TestSchema.id2,
    >>>          TestSchema.sensor_name],
    >>> }
    >>> delta_threshold = 4
    >>> timestamp_field = 'id'

      - The data being:

    >>> A {'id': 0, .....}
    >>> B {'id': 3, .....}
    >>> C {'id': 8, .....}
    >>> D {'id': 10, .....}
    >>> E {'id': 11, .....}
    >>> G {'id': 20, .....}
    >>> H {'id': 30, .....}

      - The result will be:

    >>> {-1: A, 0: B},
    >>> {-1: C, 0: D},
    >>> {-1: D, 0: E}

      - Notice how:
        - ``(B, C)`` was skipped since B ``id`` is 3 and C ``id`` is 8 (difference of 5 ``>= delta_threshold`` of 4)
        - ``(E, G), (G, H)`` were also skipped due to the same reason

    One caveat to note: All NGrams within the same parquet row group are guaranteed to be returned, but not across
    different parquet row groups. i.e. if row group 1 has ``[0, 5]``, row group 2 has ``[6, 10]`` then this will result
    in ``(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9), (9, 10)``.

    Notice how the ``(5, 6)`` was skipped because it is across two different row groups.
    In order to potentially produce more NGrams, the row group size should be increased (at minimum it needs to be
    at least as large as the NGram length).

    Note: Passing a field argument like ``{-1: [TestSchema.id], 1: [TestSchema.id]}`` are allowed, the 0 field will
    just be empty.
    Passing a field argument like ``{1: [TestSchema.id], 0: [TestSchema.id]}`` is the same as passing a field
    argument like ``{0: [TestSchema.id], 1: [TestSchema.id]}``

    The return type will be a dictionary where the keys are the same as the keys passed to fields and the value of each
    key will be the item.
    """

    def __init__(self, fields, delta_threshold, timestamp_field, timestamp_overlap=True):
        """
        Constructor to initialize ngram with fields, delta_threshold and timestamp_field.

        :param fields: A dictionary, with consecutive integers as keys and each value is an array of Unischema fields.
        :param delta_threshold: The maximum threshold of delta between timestamp_field.
        :param timestamp_field: The field that represents the timestamp.
        :param timestamp_overlap: Whether timestamps in sequences are allowed to overlap (defaults to True),
            e.g., If the data consists of consecutive timestamps ``[{'id': 0}, {'id': 1}, ..., {'id': 5}]``
            and you are asking for NGram of length 3 with timestamp_overlap set to True you will receive
            NGrams of ``[{'id': 0}, {'id': 1}, {'id': 2}]`` and ``[{'id': 1}, {'id': 2}, {'id': 3}]`` (in addition
            to others); however, note that ``{'id': 1}``, and ``{'id': 2}`` appear twice. With timestamp_overlap set
            to False, this would not occur and instead return ``[{'id': 0}, {'id': 1}, {'id': 2}]`` and
            ``[{'id': 3}, {'id': 4}, {'id': 5}]``. There is no overlap of timestamps between NGrams (and each
            timestamp record should only occur once in the returned data)
        """
        self._fields = fields
        self._delta_threshold = delta_threshold

        self._timestamp_field = timestamp_field

        self.timestamp_overlap = timestamp_overlap

        self._validate_ngram(fields, delta_threshold, timestamp_field, timestamp_overlap)

    @property
    def length(self):
        """
        Returns the ngram length requested.
        :return: the ngram length.
        """
        return max(self._fields.keys()) - min(self._fields.keys()) + 1

    @property
    def fields(self):
        """
        Returns the ngram fields.
        :return: The ngram fields.
        """
        return self._fields

    @property
    def delta_threshold(self):
        """
        The maximum difference between one entry and the following one in timestamp_field.
        :return: The delta threshold.
        """
        return self._delta_threshold

    def _validate_ngram(self, fields, delta_threshold, timestamp_field, timestamp_overlap):
        """
        Validates the fields, delta_threshold and timestamp_field are set and of the correct types.
        :param fields: The ngram fields.
        :param delta_threshold: The delta threshold.
        :param timestamp_field: The timestamp field.
        :param timestamp_overlap: Whether timestamps in sequences are allowed to overlap
        """
        if fields is None or not isinstance(fields, dict):
            raise ValueError('Fields must be set and must be a dictionary.')

        for key in fields:
            if not isinstance(fields[key], list):
                raise ValueError('Each field value must be a list of unischema fields/regular expression(s)')
            for field in fields[key]:
                if not (isinstance(field, UnischemaField) or isinstance(field, string_types)):
                    raise ValueError('All field values must be of type UnischemaField/regular expression')

        if delta_threshold is None or not isinstance(delta_threshold, numbers.Number):
            raise ValueError('delta_threshold must be a number.')

        if timestamp_field is None or not (isinstance(timestamp_field, UnischemaField) or
                                           isinstance(timestamp_field, string_types)):
            raise ValueError('timestamp_field must be set and must be of type UnischemaField or regular expression')

        if timestamp_overlap is None or not isinstance(timestamp_overlap, bool):
            raise ValueError('timestamp_overlap must be set and must be of type bool')

    def _ngram_pass_threshold(self, ngram):
        """
        Returns true if each item in a ngram passes the threshold, otherwise False.
        Specifically that means that timestamp of an item - timestamp of previous item <= delta_threshold

        It is assumed here that items read are all sorted by timestamp field.

        :param ngram: An array of items
        :return: True if each item in a ngram passes threshold, otherwise False.
        """
        # Verify that each element and it's previous element do not exceed the delta_threshold
        for previous, current in zip(ngram[:-1], ngram[1:]):
            if current[self._timestamp_field.name] - previous[self._timestamp_field.name] > self.delta_threshold:
                return False
        return True

    def resolve_regex_field_names(self, schema):
        """Resolve string(s) (regular expression(s)) that were passed into 'fields' and 'timestamp_field' parameters
        """
        self._fields = {k: self.convert_fields(schema, self._fields[k]) for k in self._fields.keys()}
        converted_ts_field = self.convert_fields(schema, [self._timestamp_field])
        if len(converted_ts_field) > 1:
            raise ValueError("timestamp_field was matched to more than one unischema field")

        self._timestamp_field = converted_ts_field[0]

    def get_field_names_at_timestep(self, timestep):
        """
        Return the field names at a certain timestep.
        :param timestep: The timestep to return the field names at.
        :return: A list of all the field names at that timestep.
        """
        if timestep not in self._fields:
            return []
        return [field.name for field in self._fields[timestep]]

    def get_schema_at_timestep(self, schema, timestep):
        """
        Returns the schema of the data at a certain timestep.
        :param schema: The schema of the data, which schema at a certain timestep is a subset of.
        :param timestep: The timestep to get the schema at.
        :return: The schema of the data at a certain timestep.
        """
        return schema.create_schema_view([schema.fields.get(field) for field in schema.fields if
                                          field in self.get_field_names_at_timestep(timestep)])

    def form_ngram(self, data, schema):
        """
        Return all the ngrams as dictated by fields, delta_threshold and timestamp_field.
        :param data: The data items, which is a list of Unischema items.
        :return: A dictionary, with keys ``[0, length - 1]``. The value of each key is the corresponding item in the
        ngram at that position.
        """

        base_key = min(self._fields.keys())
        result = []
        prev_ngram_end_timestamp = None

        for index in range(len(data) - self.length + 1):
            # Potential ngram: [index, index + self.length[
            potential_ngram = data[index:index + self.length]

            is_sorted = all(potential_ngram[i][self._timestamp_field.name] <=
                            potential_ngram[i + 1][self._timestamp_field.name]
                            for i in range(len(potential_ngram) - 1))
            if not is_sorted:
                raise NotImplementedError('NGram assumes that the data is sorted by {0} field which is not the case'
                                          .format(self._timestamp_field.name))

            if not self.timestamp_overlap and prev_ngram_end_timestamp is not None:
                # If we dont want timestamps of NGrams to overlap, check that the start timestamp of the next NGram
                # is not less than the end timestamp of the previous NGram
                next_ngram_start_timestamp = potential_ngram[0][self._timestamp_field.name]
                if next_ngram_start_timestamp <= prev_ngram_end_timestamp:
                    continue

            # If all elements in potential_ngram passes the ngram threshold
            # (i.e. current element timestamp - previous element timestamp <= delta_threshold)
            # then add the potential ngram in the results, otherwise skip it
            if len(potential_ngram) == self.length and self._ngram_pass_threshold(potential_ngram):
                new_item = {(base_key + key): value for (key, value) in enumerate(potential_ngram)}
                for key in new_item:
                    # Get the data for that current timestep and create a namedtuple
                    current_item = new_item[key]
                    new_item[key] = {k: current_item[k]
                                     for k in current_item if k in self.get_field_names_at_timestep(key)}
                result.append(new_item)

                if not self.timestamp_overlap:
                    prev_ngram_end_timestamp = potential_ngram[-1][self._timestamp_field.name]

        return result

    def make_namedtuple(self, schema, ngram_as_dicts):
        """Converts a ngram structure where mapped values are dictionaries to a mapped structure where mapped values
        are namedtuples.

        Example:

        >>> { -1 : {'f1': 10, 'f2': 20},
        >>>    0 : {'f1': 30},
        >>> }

        is converted to

        >>> { -1 : namedtuple(f1=10, f2=20),
        >>>    0 : namedtuple(f1=30),
        >>> }

        :param schema: schema used for conversion
        :param ngram_as_dicts: ngram in the timestamp-to-dict mapping
        :return: ngram in the timestamp-to-namedtuple mapping
        """
        ngram_as_tuples = {}
        for timestamp in ngram_as_dicts.keys():
            data_as_dict = ngram_as_dicts[timestamp]
            current_schema = self.get_schema_at_timestep(schema=schema, timestep=timestamp)
            ngram_as_tuples[timestamp] = current_schema.make_namedtuple(**data_as_dict)
        return ngram_as_tuples

    def get_field_names_at_all_timesteps(self):
        """Returns a list of fields that are present at least in the one of the time offsets"""
        return list({field for fields in self._fields.values() for field in fields})

    def convert_fields(self, unischema, field_list):
        """Convert all the fields in field_list into Unischema fields.
        field_list can contain unischema fields and strings (regular expressions)

        :param unischema: Unischema object
        :param field_list: A list of unischema fields or strings (regular expressions)
        :return: list of unischema fields
        """
        # Split fields parameter to regex pattern strings and UnischemaField objects
        regex_patterns = [f for f in field_list if isinstance(f, string_types)]
        # We can not check type against UnischemaField because the artifact introduced by
        # pickling, since depickled UnischemaField are of type collections.UnischemaField
        # while withing depickling they are of petastorm.unischema.UnischemaField
        # Since UnischemaField is a tuple, we check against it since it is invariant to
        # pickling
        unischema_field_objects = [f for f in field_list if isinstance(f, tuple)]

        if len(unischema_field_objects) + len(regex_patterns) != len(field_list):
            raise ValueError('"Elements of fields"/"timestamp field" must be either a string (regular expressions) or'
                             ' an instance of UnischemaField class.')

        converted_fields = unischema_field_objects + match_unischema_fields(unischema, regex_patterns)

        return converted_fields
