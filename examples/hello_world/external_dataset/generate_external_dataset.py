#  Copyright (c) 2018-2019 Uber Technologies, Inc.
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

"""
This is part of a minimal example of how to use petastorm to read a dataset not created
with petastorm. Generates a sample dataset from random data.
"""

import random
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, IntegerType


NON_PETASTORM_SCHEMA = StructType([
    StructField("id", IntegerType(), True),
    StructField("value1", IntegerType(), True),
    StructField("value2", IntegerType(), True)
])


def row_generator(x):
    """Returns a single entry in the generated dataset. Return a bunch of random values as an example."""
    return Row(id=x, value1=random.randint(-255, 255), value2=random.randint(-255, 255))


def generate_external_dataset(output_url='file:///tmp/external_dataset'):
    """Creates an example dataset at output_url in Parquet format"""
    spark = SparkSession.builder\
        .master('local[2]')\
        .getOrCreate()
    sc = spark.sparkContext

    rows_count = 10
    rows_rdd = sc.parallelize(range(rows_count))\
        .map(row_generator)

    spark.createDataFrame(rows_rdd).\
        write.\
        mode('overwrite').\
        parquet(output_url)


if __name__ == '__main__':
    generate_external_dataset()
