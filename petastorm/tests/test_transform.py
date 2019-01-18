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

from petastorm.transform import transform_schema, TransformSpec
from petastorm.unischema import Unischema, UnischemaField

TestSchema = Unischema('TestSchema', [
    UnischemaField('string', np.unicode_, (), None, False),
    UnischemaField('int', np.int32, (), None, False),
    UnischemaField('double', np.float64, (), None, False),
])


def test_noop_transform():
    transformed_schema = transform_schema(TestSchema, TransformSpec(lambda x: x, edit_fields=None, removed_fields=None))
    assert transformed_schema.fields == TestSchema.fields


def test_remove_field_transform():
    one_removed = transform_schema(TestSchema, TransformSpec(lambda x: x, edit_fields=None,
                                                             removed_fields=['int']))
    assert set(one_removed.fields.keys()) == {'string', 'double'}

    two_removed = transform_schema(TestSchema, TransformSpec(lambda x: x, edit_fields=None,
                                                             removed_fields=['int', 'double']))
    assert set(two_removed.fields.keys()) == {'string'}


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
