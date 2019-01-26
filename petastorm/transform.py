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

from petastorm.unischema import UnischemaField, Unischema


class TransformSpec(object):
    def __init__(self, func, edit_fields=None, removed_fields=None):
        """TransformSpec defines a user transformation that is applied to a loaded row on a worker thread/process.

        The object defines the function (callable) that perform the transform as well as the
        schema transform: pre-transform-schema to post-transform-schema.

        ``func`` argument is a callable which takes a row as its parameter and returns a modified row.
        ``edit_fields`` and ``removed_fields`` define mutating operations performed on the original schema that
        produce a post-transform schema. ``func`` return value must comply to this post-transform schema.

        :param func: A callable. The function is called on the worker thread. It takes a dictionary that complies to
          the input schema and must return a dictionary that complies to a post-transform schema.
        :param edit_fields: Optional. A list of 4-tuples with the following fields:
          ``(name, numpy_dtype, shape, is_nullable)``
        :param removed_fields: Optional. A list of field names that will be removed from the original schema.
        """
        self.func = func
        self.edit_fields = edit_fields or []
        self.removed_fields = removed_fields or []


def transform_schema(schema, transform_spec):
    """Creates a post-transform given a pre-transform schema and a transform_spec with mutation instructions.

    :param schema: A pre-transform schema
    :param transform_spec: a TransformSpec object with mutation instructions.
    :return: A post-transform schema
    """
    removed_fields = set(transform_spec.removed_fields)
    unknown_field_names = removed_fields - set(schema.fields.keys())
    if unknown_field_names:
        raise ValueError('Unexpected field names found in TransformSpec remove_fields list: "%s". '
                         'Valid values are "%s".',
                         ', '.join(removed_fields), ', '.join(schema.fields.keys()))

    exclude_fields = {f[0] for f in transform_spec.edit_fields} | removed_fields
    fields = [v for k, v in schema.fields.items() if k not in exclude_fields]

    for field_to_edit in transform_spec.edit_fields:
        edited_unischema_field = UnischemaField(name=field_to_edit[0], numpy_dtype=field_to_edit[1],
                                                shape=field_to_edit[2], codec=None, nullable=field_to_edit[3])
        fields.append(edited_unischema_field)

    return Unischema(schema.name + '_transformed', fields)
