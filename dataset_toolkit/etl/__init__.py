#
# Uber, Inc. (c) 2018
#

import abc


class RowGroupIndexerBase(object):
    """ Base class for row group indexers."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __add__(self, other):
        pass

    @abc.abstractproperty
    def index_name(self):
        """ Return unique index name."""
        return None

    @abc.abstractproperty
    def column_names(self):
        """ Return list of column(s) reuired to build index."""
        return None

    @abc.abstractproperty
    def indexed_values(self):
        """ Return list of values in index"""
        return None

    @abc.abstractmethod
    def get_row_group_indexes(self, value_key):
        """ Return row groups for given value in index."""
        return None

    @abc.abstractmethod
    def build_index(self, decoded_rows, piece_index):
        """ index values in given rows."""
        pass
