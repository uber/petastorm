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

"""In order for the predicates to be accessible from a process_pool and the test_end_to_end.py is ran directly as
__main__, these predicates have to be implemented in a separate module"""
from petastorm.predicates import PredicateBase


class PartitionKeyInSetPredicate(PredicateBase):
    def __init__(self, inclusion_values):
        self._inclusion_values = inclusion_values

    def get_fields(self):
        return {'partition_key'}

    def do_include(self, values):
        return values['partition_key'] in self._inclusion_values


class EqualPredicate(PredicateBase):
    def __init__(self, values):
        self._values = values

    def get_fields(self):
        return list(self._values.keys())

    def do_include(self, values):
        return self._values == values
