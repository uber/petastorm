#
# Uber, Inc. (c) 2018
#

from dataset_toolkit import RowGroupSelectorBase


class SingleIndexSelector(RowGroupSelectorBase):
    """
    Generic selector for single field indexer.
    Select all row groups containing any of given values.
    """

    def __init__(self, index_name, values_list):
        self._index_name = index_name
        self._values_to_select = values_list

    def get_index_names(self):
        return [self._index_name]

    def select_row_groups(self, index_dict):
        indexer = index_dict[self._index_name]
        row_groups = set()
        for value in self._values_to_select:
            row_groups |= indexer.get_row_group_indexes(value)
        return row_groups
