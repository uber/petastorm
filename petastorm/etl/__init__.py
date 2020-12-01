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
class RowGroupIndexerBase(object):
    """ Base class for row group indexers."""

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
