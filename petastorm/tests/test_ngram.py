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

from petastorm.ngram import NGram
from petastorm.unischema import UnischemaField, Unischema

TestSchema = Unischema('TestSchema', [
    UnischemaField('string', np.unicode_, (), None, False),
    UnischemaField('int', np.int32, (), None, False),
    UnischemaField('double', np.float64, (), None, False),
])


def test_eq():
    ngram1 = NGram({-1: [TestSchema.string], 0: [TestSchema.int]}, delta_threshold=1, timestamp_field=TestSchema.int)
    ngram2 = NGram({-1: [TestSchema.string], 0: [TestSchema.int]}, delta_threshold=1, timestamp_field=TestSchema.int)

    assert ngram1 == ngram1
    assert ngram1 == ngram2

    assert not ngram1 != ngram1
    assert not ngram1 != ngram2

    ngram3 = NGram({0: [TestSchema.string], 2: [TestSchema.int]}, delta_threshold=1, timestamp_field=TestSchema.int)
    assert ngram1 != ngram3
    assert not ngram1 == ngram3

    ngram4 = NGram({-1: [TestSchema.int], 0: [TestSchema.int]}, delta_threshold=1, timestamp_field=TestSchema.int)
    assert ngram1 != ngram4
    assert not ngram1 == ngram4
