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

from petastorm import utils


class RowDecoder(object):
    def __init__(self, schema, ngram):
        """RowDecoder decodes a row with encoded data loaded from a dataset and applies a decoder on each field.

        :param schema: A Unischema instance
        :param ngram: An NGram object instance. `ngram=None` indicates row is not an ngram
        """
        self._schema = schema
        self._ngram = ngram

    def decode(self, row):
        """Decodes each value in the dictionary as defined by Unischema (passed to the constructor). Returns a
        namedtuple

        Example (ngram is None):

        Input:
          row = {
            'field1': bytearray(),
            ...
          }

        Output:
          (
            field1=numpy.array(...),
            ...
          )


        Example (ngram):

        row = {
          -1 : { 'field1': bytearray(), ...},
          0 : { 'field2': bytearray(), ...},
          1 : { 'field1': bytearray(), ...},
        }

        Output:

        row = {
          -1 : ( field1=numpy.array(...), ...),
          0  : ( field2=numpy.array(...), ...),
          1  : ( field1=numpy.array(...), ...),
        }


        :param row: A dictionary of fields
        :return: A named tuple (or a dictionary of namedtuples in case of a non None ngram)
        """

        if self._ngram:
            for key in row.keys():
                current_schema = self._ngram.get_schema_at_timestep(self._schema, key)
                row[key] = current_schema.make_namedtuple(**utils.decode_row(row[key], current_schema))
            return row
        else:
            return self._schema.make_namedtuple(**utils.decode_row(row, self._schema))
