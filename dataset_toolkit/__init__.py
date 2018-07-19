#
# Uber, Inc. (c) 2018
#
import abc

__version__ = '0.1.2'

class PredicateBase(object):
    """ Base class for row predicates """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_fields(self):
        pass

    @abc.abstractmethod
    def do_include(self):
        pass


class RowGroupSelectorBase(object):
    """ Base class for row group selectors."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_index_names(self):
        """ Return list of indexes required for given selector."""
        pass

    @abc.abstractmethod
    def select_row_groups(self, index_dict):
        """ Return set of row groups which are selected."""
        pass
