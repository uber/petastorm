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

"""Script to add petastorm metadata to an existing parquet dataset"""

import argparse
import sys
from pydoc import locate

from pyarrow import parquet as pq
from pyspark.sql import SparkSession

from petastorm.etl.dataset_metadata import materialize_dataset, get_schema, ROW_GROUPS_PER_FILE_KEY
from petastorm.etl.rowgroup_indexing import ROWGROUPS_INDEX_KEY
from petastorm.fs_utils import FilesystemResolver
from petastorm.utils import add_to_dataset_metadata

example_text = '''This is meant to be run as a spark job. Example (some replacement required):

On Spark:    spark-submit \
    --master spark://ip:port \
    $(which petastorm-generate-metadata) \
    --dataset_url hdfs:///path/to/my/hello_world_dataset \
    --unischema_class examples.hello_world.hello_world_dataset.HelloWorldSchema

Locally:     petastorm-generate-metadata \
    --dataset_url hdfs:///path/to/my/hello_world_dataset \
    --unischema_class examples.hello_world.hello_world_dataset.HelloWorldSchema
    --master local[*]
'''


def generate_petastorm_metadata(spark, dataset_url, unischema_class=None):
    """
    Generate metadata necessary to read a petastorm dataset to an existing dataset.
    :param spark: spark session
    :param dataset_url: url of existing dataset
    :param unischema_class: (optional) fully qualified dataset unischema class. If not specified will attempt
        to find one already in the dataset. (e.g. examples.hello_world.hello_world_dataset.HelloWorldSchema)
    :return:
    """
    sc = spark.sparkContext

    resolver = FilesystemResolver(dataset_url, sc._jsc.hadoopConfiguration())
    dataset = pq.ParquetDataset(
        resolver.parsed_dataset_url().path,
        filesystem=resolver.filesystem(),
        validate_schema=False)

    if unischema_class:
        schema = locate(unischema_class)
    else:

        try:
            schema = get_schema(dataset)
        except ValueError:
            raise ValueError('Unischema class could not be located in existing dataset,'
                             ' please specify it')

    # In order to be backwards compatible, we retrieve the common metadata from the dataset before
    # overwriting the metadata to keep row group indexes and the old row group per file index
    arrow_metadata = dataset.common_metadata or None

    with materialize_dataset(spark, dataset_url, schema):
        # Inside the materialize dataset context we just need to write the metadata file as the schema will
        # be written by the context manager.
        # We use the java ParquetOutputCommitter to write the metadata file for the existing dataset
        # which will read all the footers of the dataset in parallel and merge them.
        hadoop_config = sc._jsc.hadoopConfiguration()
        Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
        parquet_output_committer = sc._gateway.jvm.org.apache.parquet.hadoop.ParquetOutputCommitter
        parquet_output_committer.writeMetaDataFile(hadoop_config, Path(dataset_url))

    if arrow_metadata:
        # If there was the old row groups per file key or the row groups index key, add them to the new dataset metadata
        base_schema = arrow_metadata.schema.to_arrow_schema()
        metadata_dict = base_schema.metadata
        if ROW_GROUPS_PER_FILE_KEY in metadata_dict:
            add_to_dataset_metadata(dataset, ROW_GROUPS_PER_FILE_KEY, metadata_dict[ROW_GROUPS_PER_FILE_KEY])
        if ROWGROUPS_INDEX_KEY in metadata_dict:
            add_to_dataset_metadata(dataset, ROWGROUPS_INDEX_KEY, metadata_dict[ROWGROUPS_INDEX_KEY])


def _main(args):
    parser = argparse.ArgumentParser(prog='petastorm_generate_metadata',
                                     description='Add necessary petastorm metadata to an existing dataset',
                                     epilog=example_text,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset_url',
                        help='the url to the dataset base directory', required=True)
    parser.add_argument('--unischema_class',
                        help='the fully qualified class of the dataset unischema. If not specified will attempt'
                             ' to reuse schema already in dataset. '
                             '(e.g. examples.hello_world.hello_world_dataset.HelloWorldSchema)', required=False)
    parser.add_argument('--master', type=str,
                        help='Spark master. Default if not specified. To run on a local machine, specify '
                             '"local[W]" (where W is the number of local spark workers, e.g. local[10])')
    args = parser.parse_args(args)

    # Open Spark Session
    spark_session = SparkSession \
        .builder \
        .appName("Petastorm Metadata Index")
    if args.master:
        spark_session.master(args.master)

    spark = spark_session.getOrCreate()

    generate_petastorm_metadata(spark, args.dataset_url, args.unischema_class)

    # Shut down the spark sessions and context
    spark.sparkContext.stop()


def main():
    _main(sys.argv[1:])


if __name__ == '__main__':
    _main(sys.argv[1:])
