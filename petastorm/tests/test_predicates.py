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

import pytest

from petastorm.predicates import in_set, in_intersection, \
    in_negate, in_reduce, in_pseudorandom_split, in_lambda
from petastorm.reader import Reader
from petastorm.tests.test_common import TestSchema
from petastorm.workers_pool.dummy_pool import DummyPool


@pytest.fixture(scope="session")
def all_values(request, tmpdir_factory):
    all_values = set()
    for i in range(10000):
        all_values.add('guid_' + str(i))
    return all_values


def test_inclusion(all_values):
    for values in [{'guid_2', 'guid_1'}, {'guid_5', 'guid_XXX'}, {'guid_2'}]:
        test_predicate = in_set(values, 'volume_guid')
        included_values = set()
        for val in all_values:
            if test_predicate.do_include({'volume_guid': val}):
                included_values.add(val)
        assert included_values == all_values.intersection(values)


def test_list_inclusion(all_values):
    for values in [{'guid_2', 'guid_1'}, {'guid_5', 'guid_XXX'}, {'guid_XX'}]:
        test_predicate = in_intersection(values, 'volume_guid')
        included = test_predicate.do_include({'volume_guid': list(all_values)})
        assert included != all_values.intersection(values)


def test_custom_function(all_values):
    for value in ['guid_2', 'guid_1', 'guid_5', 'guid_XXX', 'guid_XX']:
        test_predicate = in_lambda(['volume_guids'], lambda volume_guids, val=value: val in volume_guids)
        included = test_predicate.do_include({'volume_guids': all_values})
        assert included == (value in all_values)


def test_custom_function_with_state(all_values):
    counter = [0]

    def pred_func(volume_guids, cntr):
        cntr[0] += 1
        return volume_guids in all_values

    test_predicate = in_lambda(['volume_guids'], pred_func, counter)
    for value in ['guid_2', 'guid_1', 'guid_5', 'guid_XXX', 'guid_XX']:
        included = test_predicate.do_include({'volume_guids': value})
        assert included == (value in all_values)
    assert counter[0] == 5


def test_negation(all_values):
    for values in [{'guid_2', 'guid_1'}, {'guid_5', 'guid_9'}, {'guid_2'}]:
        test_predicate = in_negate(in_set(values, 'volume_guid'))
        included_values = set()
        for val in all_values:
            if test_predicate.do_include({'volume_guid': val}):
                included_values.add(val)
        assert included_values == all_values.difference(values)


def test_and_argegarion(all_values):
    for values1 in [{'guid_0', 'guid_1'}, {'guid_3', 'guid_6', 'guid_20'}, {'guid_2'}]:
        for values2 in [{'guid_2', 'guid_1'}, {'guid_5', 'guid_9'}, {'guid_2'}]:
            test_predicate = in_reduce(
                [in_set(values1, 'volume_guid'), in_set(values2, 'volume_guid')], all)
            included_values = set()
            for val in all_values:
                if test_predicate.do_include({'volume_guid': val}):
                    included_values.add(val)
            assert included_values == values1.intersection(values2)


def test_or_argegarion(all_values):
    for values1 in [{'guid_0', 'guid_1'}, {'guid_3', 'guid_6', 'guid_20'}, {'guid_2'}]:
        for values2 in [{'guid_2', 'guid_1'}, {'guid_5', 'guid_9'}, {'guid_2'}]:
            test_predicate = in_reduce(
                [in_set(values1, 'volume_guid'), in_set(values2, 'volume_guid')], any)
            included_values = set()
            for val in all_values:
                if test_predicate.do_include({'volume_guid': val}):
                    included_values.add(val)
            assert included_values == values1.union(values2)


def test_pseudorandom_split(all_values):
    split_list = [0.3, 0.4, 0.1, 0.0, 0.2]
    values_num = len(all_values)
    for idx in range(len(split_list)):
        test_predicate = in_pseudorandom_split(split_list, idx, 'volume_guid')
        included_values = set()
        for val in all_values:
            if test_predicate.do_include({'volume_guid': val}):
                included_values.add(val)
        expected_num = values_num * split_list[idx]
        assert pytest.approx(len(included_values), expected_num * 0.1) == expected_num


def test_predicate_on_single_column(synthetic_dataset):
    reader = Reader(synthetic_dataset.url,
                    schema_fields=[TestSchema.id2],
                    predicate=in_lambda(['id2'], lambda id2: True),
                    reader_pool=DummyPool())
    counter = 0
    for row in reader:
        counter += 1
        actual = dict(row._asdict())
        assert actual['id2'] < 2
    assert counter == len(synthetic_dataset.data)
