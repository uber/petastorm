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

"""
This is a minimal example of how to generate a petastorm dataset. Generates a
sample dataset with some random data.
"""
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField

# The schema defines how the dataset schema looks like
HelloWorldSchema = Unischema('HelloWorldSchema', [
    UnischemaField('image1', np.uint8, (300, 100, 3), CompressedImageCodec('png'), False),
])


def row_generator(x):
    """Returns a single entry in the generated dataset. Return a bunch of random values as an example."""
    return {'image1': np.random.randint(0, 255, dtype=np.uint8, size=(300, 100, 3))}


def generate_petastorm_dataset(output_url='file:///tmp/hello_world_dataset'):
    rowgroup_size_mb = 256

    spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()
    sc = spark.sparkContext

    # Wrap dataset materialization portion. Will take care of setting up spark environment variables as
    # well as save petastorm specific metadata
    rows_count = 1000
    with materialize_dataset(spark, output_url, HelloWorldSchema, rowgroup_size_mb):

        rows_rdd = sc.parallelize(range(rows_count))\
            .map(row_generator)\
            .map(lambda x: dict_to_spark_row(HelloWorldSchema, x))

        spark.createDataFrame(rows_rdd, HelloWorldSchema.as_spark_schema()) \
            .coalesce(10) \
            .write \
            .mode('overwrite') \
            .parquet(output_url)


if __name__ == '__main__':
    from PIL import Image
    from io import BytesIO
    from pathlib import Path


    output_path = "/tmp/thousand_images_dataset"
    generate_petastorm_dataset("file://" + output_path)

    root_directory = Path(output_path)
    parquet_size = sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file())
    print(f"Parquet size {parquet_size / 2**10} KB")

    image = row_generator(0)["image1"]
    buffer = BytesIO()
    im = Image.fromarray(image)
    im.save(buffer, format="png")
    single_png_size = len(buffer.getbuffer())
    print(f"png file size: {single_png_size/2**10} KB")
    print(f"Size per parquet row: {parquet_size / 1000 / 2**10} KB")

