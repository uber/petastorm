#
# Uber, Inc. (c) 2018
#
"""In order for the predicates to be accessible from a process_pool and the test_end_to_end.py is ran directly as
__main__, these predicates have to be implemented in a separate module"""
from dataset_toolkit import PredicateBase


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
        return self._values.keys()

    def do_include(self, values):
        return self._values == values
