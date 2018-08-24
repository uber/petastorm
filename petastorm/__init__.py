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

__version__ = '0.3.1'


@six.add_metaclass(abc.ABCMeta)
class PredicateBase(object):
    """ Base class for row predicates """

    @abc.abstractmethod
    def get_fields(self):
        pass

    @abc.abstractmethod
    def do_include(self, values):
        pass


@six.add_metaclass(abc.ABCMeta)
class RowGroupSelectorBase(object):
    """ Base class for row group selectors."""

    @abc.abstractmethod
    def get_index_names(self):
        """ Return list of indexes required for given selector."""
        pass

    @abc.abstractmethod
    def select_row_groups(self, index_dict):
        """ Return set of row groups which are selected."""
        pass
