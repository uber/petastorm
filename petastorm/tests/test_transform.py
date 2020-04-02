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

import numpy as np
import pytest

from petastorm.transform import edit_field, transform_schema, TransformSpec
from petastorm.unischema import Unischema, UnischemaField

TestSchema = Unischema('TestSchema', [
    UnischemaField('string', np.unicode_, (), None, False),
    UnischemaField('int', np.int32, (), None, False),
    UnischemaField('double', np.float64, (), None, False),
])


def test_noop_transform():
    transformed_schema = transform_schema(TestSchema, TransformSpec(lambda x: x, edit_fields=None, removed_fields=None))
    assert set(transformed_schema.fields) == set(TestSchema.fields)


def test_remove_field_transform():
    one_removed = transform_schema(TestSchema, TransformSpec(lambda x: x, edit_fields=None,
                                                             removed_fields=['int']))
    assert set(one_removed.fields.keys()) == {'string', 'double'}

    two_removed = transform_schema(TestSchema, TransformSpec(lambda x: x, edit_fields=None,
                                                             removed_fields=['int', 'double']))
    assert set(two_removed.fields.keys()) == {'string'}


def test_select_field_transform():
    test_list = [
        ['string', 'double', 'int'],
        ['int', 'string', 'double'],
        ['string', 'int'],
        ['int']
    ]
    for selected_fields in test_list:
        transformed = transform_schema(TestSchema, TransformSpec(selected_fields=selected_fields))
        assert list(transformed.fields.keys()) == selected_fields


def test_add_field_transform():
    one_added = transform_schema(TestSchema,
                                 TransformSpec(lambda x: x,
                                               edit_fields=[UnischemaField('double2', np.float64, (), None, False)]))
    assert set(one_added.fields.keys()) == {'string', 'double', 'double2', 'int'}


def test_change_field_transform():
    one_added = transform_schema(TestSchema,
                                 TransformSpec(lambda x: x,
                                               edit_fields=[UnischemaField('double', np.float16, (), None, False)]))
    assert one_added.fields['double'].numpy_dtype == np.float16


def test_unknown_fields_in_remove_field_transform():
    with pytest.warns(UserWarning, match='not part of the schema.*unknown_1'):
        one_removed = transform_schema(TestSchema, TransformSpec(lambda x: x, edit_fields=None,
                                                                 removed_fields=['int', 'unknown_1', 'unknown_2']))
    assert set(one_removed.fields.keys()) == {'string', 'double'}


def test_create_edit_field():
    e1 = edit_field(name='ab', numpy_dtype=np.float64, shape=(2, 3), nullable=True)
    assert e1 == ('ab', np.float64, (2, 3), True)
