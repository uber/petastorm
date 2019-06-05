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
import pyarrow
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from petastorm.codecs import ScalarCodec
from petastorm.etl.dataset_metadata import get_schema_from_dataset_url, materialize_dataset
from petastorm.tests.test_common import TestSchema
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row


def test_get_schema_from_dataset_url(synthetic_dataset):
    schema = get_schema_from_dataset_url(synthetic_dataset.url)
    assert TestSchema.fields == schema.fields


def test_get_schema_from_dataset_url_bogus_url():
    with pytest.raises(IOError):
        get_schema_from_dataset_url('file:///non-existing-path')

    with pytest.raises(ValueError):
        get_schema_from_dataset_url('/invalid_url')


def test_serialize_filesystem_factory(tmpdir):
    SimpleSchema = Unischema('SimpleSchema', [
        UnischemaField('id', np.int32, (), ScalarCodec(IntegerType()), False),
        UnischemaField('foo', np.int32, (), ScalarCodec(IntegerType()), False),
    ])

    class BogusFS(pyarrow.LocalFileSystem):
        def __getstate__(self):
            raise RuntimeError("can not serialize")

    rows_count = 10
    output_url = "file://{0}/fs_factory_test".format(tmpdir)
    rowgroup_size_mb = 256
    spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()
    sc = spark.sparkContext
    with materialize_dataset(spark, output_url, SimpleSchema, rowgroup_size_mb, filesystem_factory=BogusFS):
        rows_rdd = sc.parallelize(range(rows_count))\
            .map(lambda x: {'id': x, 'foo': x})\
            .map(lambda x: dict_to_spark_row(SimpleSchema, x))

        spark.createDataFrame(rows_rdd, SimpleSchema.as_spark_schema()) \
            .write \
            .parquet(output_url)
