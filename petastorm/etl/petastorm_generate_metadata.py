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
from petastorm.unischema import Unischema
from petastorm.utils import add_to_dataset_metadata

example_text = '''Example (some replacement required):

Locally:
petastorm-generate-metadata.py \\
    --dataset_url hdfs:///path/to/my/hello_world_dataset \\
    --unischema_class examples.hello_world.generate_hello_world_dataset.HelloWorldSchema \\
    --master local[*]

On Spark:
spark-submit \\
    --master spark://ip:port \\
    $(which petastorm-generate-metadata.py) \\
    --dataset_url hdfs:///path/to/my/hello_world_dataset \\
    --unischema_class examples.hello_world.generate_hello_world_dataset.HelloWorldSchema
'''


def generate_petastorm_metadata(spark, dataset_url, unischema_class=None, use_summary_metadata=False,
                                hdfs_driver='libhdfs3'):
    """
    Generates metadata necessary to read a petastorm dataset to an existing dataset.

    :param spark: spark session
    :param dataset_url: url of existing dataset
    :param unischema_class: (optional) fully qualified dataset unischema class. If not specified will attempt
        to find one already in the dataset. (e.g.
        :class:`examples.hello_world.generate_hello_world_dataset.HelloWorldSchema`)
    :param hdfs_driver: A string denoting the hdfs driver to use (if using a dataset on hdfs). Current choices are
        libhdfs (java through JNI) or libhdfs3 (C++)
    :param user: String denoting username when connecting to HDFS
    """
    sc = spark.sparkContext

    resolver = FilesystemResolver(dataset_url, sc._jsc.hadoopConfiguration(), hdfs_driver=hdfs_driver,
                                  user=spark.sparkContext.sparkUser())
    fs = resolver.filesystem()
    dataset = pq.ParquetDataset(
        resolver.get_dataset_path(),
        filesystem=fs,
        validate_schema=False)

    if unischema_class:
        schema = locate(unischema_class)
        if not isinstance(schema, Unischema):
            raise ValueError('The specified class %s is not an instance of a petastorm.Unischema object.',
                             unischema_class)
    else:

        try:
            schema = get_schema(dataset)
        except ValueError:
            raise ValueError('Unischema class could not be located in existing dataset,'
                             ' please specify it')

    # In order to be backwards compatible, we retrieve the common metadata from the dataset before
    # overwriting the metadata to keep row group indexes and the old row group per file index
    arrow_metadata = dataset.common_metadata or None

    with materialize_dataset(spark, dataset_url, schema, use_summary_metadata=use_summary_metadata,
                             filesystem_factory=resolver.filesystem_factory()):
        if use_summary_metadata:
            # Inside the materialize dataset context we just need to write the metadata file as the schema will
            # be written by the context manager.
            # We use the java ParquetOutputCommitter to write the metadata file for the existing dataset
            # which will read all the footers of the dataset in parallel and merge them.
            hadoop_config = sc._jsc.hadoopConfiguration()
            Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
            parquet_output_committer = sc._gateway.jvm.org.apache.parquet.hadoop.ParquetOutputCommitter
            parquet_output_committer.writeMetaDataFile(hadoop_config, Path(dataset_url))

    spark.stop()

    if use_summary_metadata and arrow_metadata:
        # When calling writeMetaDataFile it will overwrite the _common_metadata file which could have schema information
        # or row group indexers. Therefore we want to retain this information and will add it to the new
        # _common_metadata file. If we were using the old legacy metadata method this file wont be deleted
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
                             '(e.g. examples.hello_world.generate_hello_world_dataset.HelloWorldSchema)',
                        required=False)
    parser.add_argument('--master', type=str,
                        help='Spark master. Default if not specified. To run on a local machine, specify '
                             '"local[W]" (where W is the number of local spark workers, e.g. local[10])')
    parser.add_argument('--spark-driver-memory', type=str, help='The amount of memory the driver process will have',
                        default='4g')
    parser.add_argument('--use-summary-metadata', action='store_true',
                        help='Whether to use the parquet summary metadata format.'
                             ' Not scalable for large amounts of columns and/or row groups.')
    parser.add_argument('--hdfs-driver', type=str, default='libhdfs3',
                        help='A string denoting the hdfs driver to use (if using a dataset on hdfs). '
                             'Current choices are libhdfs (java through JNI) or libhdfs3 (C++)')
    args = parser.parse_args(args)

    # Open Spark Session
    spark_session = SparkSession \
        .builder \
        .appName("Petastorm Generate Metadata") \
        .config('spark.driver.memory', args.spark_driver_memory)
    if args.master:
        spark_session.master(args.master)

    spark = spark_session.getOrCreate()

    generate_petastorm_metadata(spark, args.dataset_url, args.unischema_class, args.use_summary_metadata,
                                hdfs_driver=args.hdfs_driver)

    # Shut down the spark sessions and context
    spark.stop()


def main():
    _main(sys.argv[1:])


if __name__ == '__main__':
    _main(sys.argv[1:])
