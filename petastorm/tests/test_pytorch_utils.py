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
from __future__ import division

import numpy as np

from petastorm import make_reader, TransformSpec
from petastorm.pytorch import DataLoader
from petastorm.tests.test_common import TestSchema

ALL_FIELDS = set(TestSchema.fields.values())
NULLABLE_FIELDS = {f for f in TestSchema.fields.values() if f.nullable}
STRING_TENSOR_FIELDS = {f for f in TestSchema.fields.values()
                        if len(f.shape) > 0 and f.numpy_dtype in (np.string_, np.unicode_)}

PYTORCH_COMPATIBLE_FIELDS = ALL_FIELDS - STRING_TENSOR_FIELDS - NULLABLE_FIELDS


def _noop_collate(alist):
    return alist


def _str_to_int(sample):
    for k, v in sample.items():
        if v is not None and isinstance(v, np.ndarray) and v.dtype.type in (np.string_, np.unicode_):
            sample[k] = np.zeros_like(v, dtype=np.int8)
    return sample


def test_basic_pytorch_dataloader(synthetic_dataset):
    with DataLoader(make_reader(synthetic_dataset.url, schema_fields=PYTORCH_COMPATIBLE_FIELDS,
                                reader_pool_type='dummy'), collate_fn=_noop_collate) as loader:
        for item in loader:
            assert len(item) == 1


def test_pytorch_dataloader_with_transform_function(synthetic_dataset):
    with DataLoader(make_reader(synthetic_dataset.url, schema_fields=ALL_FIELDS - NULLABLE_FIELDS,
                                reader_pool_type='dummy',
                                transform_spec=TransformSpec(_str_to_int)), collate_fn=_noop_collate) as loader:
        for item in loader:
            assert len(item) == 1


def test_pytorch_dataloader_batched(synthetic_dataset):
    batch_size = 10
    loader = DataLoader(
        make_reader(synthetic_dataset.url, schema_fields=PYTORCH_COMPATIBLE_FIELDS, reader_pool_type='dummy'),
        batch_size=batch_size, collate_fn=_noop_collate)
    for item in loader:
        assert len(item) == batch_size


def test_pytorch_dataloader_context(synthetic_dataset):
    reader = make_reader(synthetic_dataset.url, schema_fields=PYTORCH_COMPATIBLE_FIELDS, reader_pool_type='dummy')
    with DataLoader(reader, collate_fn=_noop_collate) as loader:
        for item in loader:
            assert len(item) == 1
