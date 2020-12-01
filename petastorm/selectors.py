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

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class RowGroupSelectorBase(object):
    """ Base class for row group selectors."""

    @abc.abstractmethod
    def get_index_names(self):
        """ Return list of indexes required for given selector."""

    @abc.abstractmethod
    def select_row_groups(self, index_dict):
        """ Return set of row groups which are selected."""


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


class IntersectIndexSelector(RowGroupSelectorBase):
    """
    Multiple single-field indexers selector.
    Select row groups containing any of the values in all given selectors.
    """

    def __init__(self, single_index_selectors):
        """
        :param single_index_selectors: List of SingleIndexSelector
        """
        self._single_index_selectors = single_index_selectors

    def get_index_names(self):
        index_names = []
        for single_index_selector in self._single_index_selectors:
            index_names.append(single_index_selector.get_index_names()[0])
        return index_names

    def select_row_groups(self, index_dict):
        row_groups = self._single_index_selectors[0].select_row_groups(index_dict)
        for single_index_selector in self._single_index_selectors:
            row_groups &= single_index_selector.select_row_groups(index_dict)
        return row_groups


class UnionIndexSelector(RowGroupSelectorBase):
    """
    Multiple single-field indexers selector.
    Select row groups containing any of the values in at least one selector.
    """

    def __init__(self, single_index_selectors):
        """
        :param single_index_selectors: List of SingleIndexSelector
        """
        self._single_index_selectors = single_index_selectors

    def get_index_names(self):
        index_names = []
        for single_index_selector in self._single_index_selectors:
            index_names.append(single_index_selector.get_index_names()[0])
        return index_names

    def select_row_groups(self, index_dict):
        row_groups = set()
        for single_index_selector in self._single_index_selectors:
            row_groups |= single_index_selector.select_row_groups(index_dict)
        return row_groups
