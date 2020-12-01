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


import glob
import os

import numpy as np
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException

from petastorm.reader import make_reader
from petastorm.tools.copy_dataset import _main, copy_dataset


@pytest.fixture()
def spark_session():
    return SparkSession.builder.appName('petastorm-copy').getOrCreate()


def test_copy_and_overwrite_cli(tmpdir, synthetic_dataset):
    target_url = 'file:///' + os.path.join(tmpdir.strpath, 'copied_data')
    _main([synthetic_dataset.url, target_url])

    with make_reader(target_url, num_epochs=1) as reader:
        for row in reader:
            actual = row._asdict()
            expected = next(d for d in synthetic_dataset.data if d['id'] == actual['id'])
            np.testing.assert_equal(actual, expected)

    with pytest.raises(AnalysisException, match='already exists'):
        _main([synthetic_dataset.url, target_url])

    _main([synthetic_dataset.url, target_url, '--overwrite'])


def test_copy_some_fields_with_repartition_cli(tmpdir, synthetic_dataset):
    target_path = os.path.join(tmpdir.strpath, 'copied_data')
    target_url = 'file://' + target_path
    _main([synthetic_dataset.url, target_url, '--field-regex', r'\bid\b', '--partition-count', '1'])

    # Check reparititioning
    assert 1 == len(glob.glob(os.path.join(target_path, 'part-*')))

    # Check we the regex filter worked
    with make_reader(target_url, num_epochs=1) as reader:
        assert list(reader.schema.fields.keys()) == ['id']


def test_copy_not_null_rows_cli(tmpdir, synthetic_dataset):
    target_url = 'file://' + os.path.join(tmpdir.strpath, 'copied_data')

    _main([synthetic_dataset.url, target_url, '--not-null-fields', 'string_array_nullable'])
    with make_reader(target_url, num_epochs=1) as reader:
        not_null_data = list(reader)
    assert len(not_null_data) < len(synthetic_dataset.data)


def test_bad_regex(synthetic_dataset):
    with pytest.raises(ValueError, match='do not match any fields'):
        copy_dataset(None, synthetic_dataset.url, '', ['bogus_name'], [], False, 1, 196)
