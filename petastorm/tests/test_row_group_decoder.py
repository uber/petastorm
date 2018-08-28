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
from pyspark.sql.types import IntegerType

from petastorm.codecs import ScalarCodec, NdarrayCodec
from petastorm.ngram import NGram
from petastorm.reader_impl.row_group_decoder import RowDecoder
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row

_matrix_shape = (10, 20)

TestSchema = Unischema('TestSchema', [
    UnischemaField('some_number', np.uint8, (), ScalarCodec(IntegerType()), False),
    UnischemaField('some_matrix', np.float32, _matrix_shape, NdarrayCodec(), False),
])


def _rand_row(some_number=np.random.randint(0, 255)):
    row_as_dict = {
        'some_number': some_number,
        'some_matrix': np.random.random(size=_matrix_shape).astype(np.float32),
    }

    return row_as_dict


def test_row_decoding():
    expected_row = _rand_row()
    encoded_row = dict_to_spark_row(TestSchema, expected_row).asDict()

    decoder = RowDecoder(TestSchema, None)
    actual_row = decoder.decode(encoded_row)._asdict()

    # Out-of-the-box pytest `assert actual_row == expected_row` can not compare dictionaries properly
    np.testing.assert_equal(actual_row, expected_row)


def test_ngram_decoding():
    N = 5
    ngram_spec = NGram({
        -1: [TestSchema.some_number, TestSchema.some_matrix],
        0: [TestSchema.some_number],
        1: [TestSchema.some_number, TestSchema.some_matrix],
    }, 2, TestSchema.some_number)

    expected_rows = [_rand_row(n) for n in range(N)]
    encoded_rows = [dict_to_spark_row(TestSchema, row).asDict() for row in expected_rows]
    encoded_ngrams = ngram_spec.form_ngram(encoded_rows, TestSchema)

    decoder = RowDecoder(TestSchema, ngram_spec)

    # decoded_ngrams is a list of 3 dictionaries, each have -1, 0, 1 keys.
    decoded_ngrams = [decoder.decode(encoded) for encoded in encoded_ngrams]

    # Verify we got 3 dictionaries
    assert 3 == len(decoded_ngrams)

    single_sample = decoded_ngrams[0]

    # A single decoded ngram looks like this:
    #   -1: some_number, some_matrix
    #    0: some_number
    #    1: some_number, some_matrix
    assert 2 == len(single_sample[-1])
    assert 0 == single_sample[-1].some_number

    assert 1 == len(single_sample[0])
    assert 1 == single_sample[0].some_number

    assert 2 == len(single_sample[1])
    assert 2 == single_sample[1].some_number
