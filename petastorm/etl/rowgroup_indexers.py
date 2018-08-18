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
from collections import defaultdict

from petastorm.etl import RowGroupIndexerBase


class SingleFieldIndexer(RowGroupIndexerBase):
    """
    Class to index single field in parquet dataset.

    This indexer only indexes numpty strings, numpty integers, or numpy arrays of strings.
    """

    def __init__(self, index_name, index_field):
        self._index_name = index_name
        self._column_name = index_field
        self._index_data = defaultdict(set)

    def __add__(self, other):
        if not isinstance(other, SingleFieldIndexer):
            raise TypeError("Make sure Spark map function return the same indexer type")
        if self._column_name != other._column_name:
            raise ValueError("Make sure indexers in Spark map function index the same fields")

        for value_key in other._index_data:
            self._index_data[value_key].update(other._index_data[value_key])

        return self

    @property
    def index_name(self):
        return self._index_name

    @property
    def column_names(self):
        return [self._column_name]

    @property
    def indexed_values(self):
        return list(self._index_data.keys())

    def get_row_group_indexes(self, value_key):
        return self._index_data[value_key]

    def build_index(self, decoded_rows, piece_index):
        field_column = [row[self._column_name] for row in decoded_rows]
        if not field_column:
            raise ValueError("Cannot build index for empty rows, column '{}'"
                             .format(self._column_name))

        for field_val in field_column:
            if field_val is not None:
                # check type of field, if it is array index each array value,
                # otherwise index field value directly
                if isinstance(field_val, np.ndarray):
                    for val in field_val:
                        self._index_data[val].add(piece_index)
                else:
                    self._index_data[field_val].add(piece_index)

        return self._index_data


class FieldNotNullIndexer(RowGroupIndexerBase):
    """
    Class to index 'Not Null' condition forsingle field in parquet dataset
    """

    def __init__(self, index_name, index_field):
        self._index_name = index_name
        self._column_name = index_field
        self._index_data = set()

    def __add__(self, other):
        if not isinstance(other, FieldNotNullIndexer):
            raise TypeError("Make sure Spark map function return the same indexer type")
        if self._column_name != other._column_name:
            raise ValueError("Make sure indexers in Spark map function index the same fields")

        self._index_data.update(other._index_data)

        return self

    @property
    def index_name(self):
        return self._index_name

    @property
    def column_names(self):
        return [self._column_name]

    @property
    def indexed_values(self):
        return ['Field is Not Null']

    def get_row_group_indexes(self, value_key=None):
        return self._index_data

    def build_index(self, decoded_rows, piece_index):
        field_column = [row[self._column_name] for row in decoded_rows]
        if not field_column:
            raise ValueError("Cannot build index for empty rows, column '{}'"
                             .format(self._column_name))

        for field_val in field_column:
            if field_val is not None:
                self._index_data.add(piece_index)
                break

        return self._index_data
