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

import unittest

from petastorm.predicates import in_set, in_intersection, \
    in_negate, in_reduce, in_pseudorandom_split, in_lambda


class PredicatesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.all_values = set()
        for i in range(10000):
            cls.all_values.add('guid_' + str(i))

    @classmethod
    def tearDownClass(cls):
        pass

    def test_inclusion(self):
        for values in [{'guid_2', 'guid_1'}, {'guid_5', 'guid_XXX'}, {'guid_2'}]:
            test_predicate = in_set(values, 'volume_guid')
            included_values = set()
            for val in PredicatesTest.all_values:
                if test_predicate.do_include({'volume_guid': val}):
                    included_values.add(val)
            self.assertEqual(included_values, PredicatesTest.all_values.intersection(values))

    def test_list_inclusion(self):
        for values in [{'guid_2', 'guid_1'}, {'guid_5', 'guid_XXX'}, {'guid_XX'}]:
            test_predicate = in_intersection(values, 'volume_guid')
            included = test_predicate.do_include({'volume_guid': list(PredicatesTest.all_values)})
            self.assertEqual(included, len(PredicatesTest.all_values.intersection(values)) > 0)

    def test_custom_function(self):
        for value in ['guid_2', 'guid_1', 'guid_5', 'guid_XXX', 'guid_XX']:
            test_predicate = in_lambda(['volume_guids'], lambda volume_guids, val=value: val in volume_guids)
            included = test_predicate.do_include({'volume_guids': PredicatesTest.all_values})
            self.assertEqual(included, value in PredicatesTest.all_values)

    def test_custom_function_with_state(self):
        counter = [0]

        def pred_func(volume_guids, cntr):
            cntr[0] += 1
            return volume_guids in PredicatesTest.all_values

        test_predicate = in_lambda(['volume_guids'], pred_func, counter)
        for value in ['guid_2', 'guid_1', 'guid_5', 'guid_XXX', 'guid_XX']:
            included = test_predicate.do_include({'volume_guids': value})
            self.assertEqual(included, value in PredicatesTest.all_values)
        self.assertEqual(counter[0], 5)

    def test_negation(self):
        for values in [{'guid_2', 'guid_1'}, {'guid_5', 'guid_9'}, {'guid_2'}]:
            test_predicate = in_negate(in_set(values, 'volume_guid'))
            included_values = set()
            for val in PredicatesTest.all_values:
                if test_predicate.do_include({'volume_guid': val}):
                    included_values.add(val)
            self.assertEqual(included_values, PredicatesTest.all_values.difference(values))

    def test_and_argegarion(self):
        for values1 in [{'guid_0', 'guid_1'}, {'guid_3', 'guid_6', 'guid_20'}, {'guid_2'}]:
            for values2 in [{'guid_2', 'guid_1'}, {'guid_5', 'guid_9'}, {'guid_2'}]:
                test_predicate = in_reduce(
                    [in_set(values1, 'volume_guid'), in_set(values2, 'volume_guid')], all)
                included_values = set()
                for val in PredicatesTest.all_values:
                    if test_predicate.do_include({'volume_guid': val}):
                        included_values.add(val)
                self.assertEqual(included_values, values1.intersection(values2))

    def test_or_argegarion(self):
        for values1 in [{'guid_0', 'guid_1'}, {'guid_3', 'guid_6', 'guid_20'}, {'guid_2'}]:
            for values2 in [{'guid_2', 'guid_1'}, {'guid_5', 'guid_9'}, {'guid_2'}]:
                test_predicate = in_reduce(
                    [in_set(values1, 'volume_guid'), in_set(values2, 'volume_guid')], any)
                included_values = set()
                for val in PredicatesTest.all_values:
                    if test_predicate.do_include({'volume_guid': val}):
                        included_values.add(val)
                self.assertEqual(included_values, values1.union(values2))

    def test_pseudorandom_split(self):
        split_list = [0.3, 0.4, 0.1, 0.0, 0.2]
        values_num = len(PredicatesTest.all_values)
        for idx in range(len(split_list)):
            test_predicate = in_pseudorandom_split(split_list, idx, 'volume_guid')
            included_values = set()
            for val in PredicatesTest.all_values:
                if test_predicate.do_include({'volume_guid': val}):
                    included_values.add(val)
            expected_num = values_num * split_list[idx]
            self.assertAlmostEqual(len(included_values), expected_num, delta=expected_num * 0.1)


if __name__ == '__main__':
    # Delegate to the test framework.
    unittest.main()
