#
# Uber, Inc. (c) 2018
#
import numpy as np
from collections import defaultdict

from dataset_toolkit.etl import RowGroupIndexerBase


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
        return self._index_data.keys()

    def get_row_group_indexes(self, value_key):
        return self._index_data[value_key]

    def build_index(self, decoded_rows, piece_index):
        field_column = [row[self._column_name] for row in decoded_rows]
        if len(field_column) == 0:
            raise ValueError("Cannot build index for empty rows, column '{}'"
                             .format(self._column_name))

        index_single_val = isinstance(field_column[0], np.string_) or isinstance(field_column[0], np.integer)
        index_list_of_vals = (isinstance(field_column[0], np.ndarray) and
                              (len(field_column[0]) == 0 or
                               isinstance(field_column[0][0], np.string_)))
        if index_single_val == index_list_of_vals:
            raise ValueError("Cannot build index for '{}' column".format(self._column_name))

        for field_val in field_column:
            if field_val is not None:
                if index_single_val:
                    self._index_data[field_val].add(piece_index)
                if index_list_of_vals:
                    for val in field_val:
                        self._index_data[val].add(piece_index)

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
        if len(field_column) == 0:
            raise ValueError("Cannot build index for empty rows, column '{}'"
                             .format(self._column_name))

        for field_val in field_column:
            if field_val is not None:
                self._index_data.add(piece_index)
                break

        return self._index_data
