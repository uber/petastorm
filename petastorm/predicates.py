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

"""
Predicates for petastorm
"""
import abc
import collections.abc
import hashlib
import numpy as np
import six
import sys


@six.add_metaclass(abc.ABCMeta)
class PredicateBase(object):
    """ Base class for row predicates """

    @abc.abstractmethod
    def get_fields(self):
        pass

    @abc.abstractmethod
    def do_include(self, values):
        pass


def _string_to_bucket(string, bucket_num):
    hash_str = hashlib.md5(string.encode('utf-8')).hexdigest()
    return int(hash_str, 16) % bucket_num


class in_set(PredicateBase):
    """ Test if predicate_field value is in inclusion_values set """

    def __init__(self, inclusion_values, predicate_field):
        self._inclusion_values = set(inclusion_values)
        self._predicate_field = predicate_field

    def get_fields(self):
        return {self._predicate_field}

    def do_include(self, values):
        return values[self._predicate_field] in self._inclusion_values


class in_intersection(PredicateBase):
    """ Test if predicate_field list contain at least one value from inclusion_values set """

    def __init__(self, inclusion_values, _predicate_field):
        self._inclusion_values = list(inclusion_values)
        self._predicate_field = _predicate_field

    def get_fields(self):
        return {self._predicate_field}

    def do_include(self, values):
        if not isinstance(values[self._predicate_field], collections.abc.Iterable):
            raise ValueError('Predicate field should have iterable type')
        return any(np.in1d(values[self._predicate_field], self._inclusion_values))


class in_lambda(PredicateBase):
    """ Wrap up custom function to be used as a predicate
        example: in_lambda(['labels_object_roles'], lambda labels_object_roles : len(labels_object_roles) > 3)
    """

    def __init__(self, predicate_fields, predicate_func, state_arg=None):
        """
        :param predicate_fields: list of fields to be used in predicate
        :param predicate_func: predicate function
               example: lambda labels_object_roles : len(labels_object_roles) > 3
        :param state_arg: additional object to keep function state. it will be passed to
               predicate_func after fields arguments ONLY if it is not None
        """
        if not isinstance(predicate_fields, list):
            raise ValueError('Predicate fields should be a list')
        self._predicate_fields = predicate_fields
        self._predicate_func = predicate_func
        self._state_arg = state_arg

    def get_fields(self):
        return set(self._predicate_fields)

    def do_include(self, values):
        args = [values[field] for field in self._predicate_fields]
        if self._state_arg is not None:
            args.append(self._state_arg)
        return self._predicate_func(*args)


class in_negate(PredicateBase):
    """ A predicate used to negate another predicate. """

    def __init__(self, predicate):
        if not isinstance(predicate, PredicateBase):
            raise ValueError('Predicate is nor derived from PredicateBase')

        self._predicate = predicate

    def get_fields(self):
        return self._predicate.get_fields()

    def do_include(self, values):
        return not self._predicate.do_include(values)


class in_reduce(PredicateBase):
    """ A predicate used to aggregate other predicates using any reduce logical operation."""

    def __init__(self, predicate_list, reduce_func):
        """ predicate_list: list of predicates
            reduce_func: function to aggregate result of all predicates in the list
            e.g. all() will implements logical 'And', any() implements logical 'Or'
        """
        check_list = [isinstance(p, PredicateBase) for p in predicate_list]
        if not all(check_list):
            raise ValueError('Predicate is nor derived from PredicateBase')
        self._predicate_list = predicate_list
        self._reduce_func = reduce_func

    def get_fields(self):
        fields = set()
        for p in self._predicate_list:
            fields |= p.get_fields()
        return fields

    def do_include(self, values):
        include_list = [p.do_include(values) for p in self._predicate_list]
        return self._reduce_func(include_list)


class in_pseudorandom_split(PredicateBase):
    """ Split dataset according to a split list based on volume_guid.
        The split is pseudorandom (can not supply the seed yet), i.e. the split outcome is always the same.
        Split is performed by hashing volume_guid uniformly to 0:1 range and returning part of full dataset
        which was hashed in given sub-range

        Example:
            'split_list = [0.5, 0.2, 0.3]' - dataset will be split on three subsets in proportion
            subset 1: 0.5 of log data
            subset 2: 0.2 of log data
            subset 3: 0.3 of log data
            Note, split is not exact, so avoid small fraction (e.g. 0.001) to avoid empty sets
    """

    def __init__(self, fraction_list, subset_index, predicate_field):
        """ split_list: a list of log fractions (real numbers in range [0:1])
            subset_index: define which subset will be used by the Reader
        """
        if subset_index >= len(fraction_list):
            raise ValueError('subset_index is out of range')
        self._predicate_field = predicate_field
        # build CDF
        subsets_high_borders = [sum(fraction_list[:i + 1]) for i in range(len(fraction_list))]
        if subset_index:
            fraction_low = subsets_high_borders[subset_index - 1]
        else:
            fraction_low = 0
        fraction_high = subsets_high_borders[subset_index]
        self._bucket_low = fraction_low * (sys.maxsize - 1)
        self._bucket_high = fraction_high * (sys.maxsize - 1)

    def get_fields(self):
        return {self._predicate_field}

    def do_include(self, values):
        if self._predicate_field not in values.keys():
            raise ValueError('Tested values does not have split key: %s' % self._predicate_field)
        bucket_idx = _string_to_bucket(str(values[self._predicate_field]), sys.maxsize)
        return self._bucket_low <= bucket_idx < self._bucket_high
