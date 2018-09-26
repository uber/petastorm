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

    def decode(self, rows):
        """Decodes all value fields of elements in the list.

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


        :param row: A list of dictionaries (non-ngram case), or a list of ngrams (where ngram is a dictionary).
        :return: A list of dictionaries, or a list of ngrams.
        """

        # TODO(yevgeni): should consider creating two different versions of decoder: for an ngram and a non-ngram case.
        if self._ngram:
            for row in rows:
                for key in row.keys():
                    current_schema = self._ngram.get_schema_at_timestep(self._schema, key)
                    row[key] = utils.decode_row(row[key], current_schema)
        else:
            rows = [utils.decode_row(row, self._schema) for row in rows]

        return rows
