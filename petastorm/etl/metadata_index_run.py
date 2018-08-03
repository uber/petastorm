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
from pydoc import locate

from pyarrow import parquet as pq
from pyspark.sql import SparkSession

from petastorm.etl.dataset_metadata import materialize_dataset, get_schema
from petastorm.fs_utils import FilesystemResolver

example_text = '''This is meant to be run as a spark job. Example (some replacement required):

On Spark:    bin/python source/spark/runner/launch_spark_cli.py --context_name robbieg-spark --context_pool HAMBURGER
    --username robbieg --job-name metadata_index_run --spark-executor-cpus 4 --restart-if-running TRUE
    --command-line '/mnt/contexts/HAMBURGER/mycontext/software/python/av/.../metadata_index_run.py
    --dataset_url hdfs:///user/robbieg/my_dataset --unischema_class av.perception.datasets.msds.schema.MsdsSchema
    --cores 24'

Locally:     bin/python source/experimental/deepdrive/python/dataset_toolkit/etl/metadata_index_run.py
    --dataset_url hdfs:///user/robbieg/my_dataset --unischema_class av.perception.datasets.msds.schema.MsdsSchema
    --master local[100]
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='metadata_index_run',
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
    args = parser.parse_args()

    dataset_url = args.dataset_url

    # Open Spark Session
    spark_session = SparkSession \
        .builder \
        .appName("Petastorm Metadata Index")
    if args.master:
        spark_session.master(args.master)

    spark = spark_session.getOrCreate()
    # Spark Context is available from the SparkSession
    sc = spark.sparkContext

    if args.unischema_class:
        schema = locate(args.unischema_class)
    else:
        resolver = FilesystemResolver(dataset_url, sc._jsc.hadoopConfiguration())
        dataset = pq.ParquetDataset(
            resolver.parsed_dataset_url().path,
            filesystem=resolver.filesystem(),
            validate_schema=False)

        try:
            schema = get_schema(dataset)
        except ValueError:
            raise ValueError('Schema could not be located in existing dataset.'
                             ' Please pass it into the job as --unischema_class')

    with materialize_dataset(spark, dataset_url, schema):
        # Inside the materialize dataset context we just need to write the metadata file as the schema will
        # be written by the context manager.
        # We use the java ParquetOutputCommitter to write the metadata file for the existing dataset
        # which will read all the footers of the dataset in parallel and merge them.
        hadoop_config = sc._jsc.hadoopConfiguration()
        Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
        parquet_output_committer = sc._gateway.jvm.org.apache.parquet.hadoop.ParquetOutputCommitter
        parquet_output_committer.writeMetaDataFile(hadoop_config, Path(args.dataset_url))

    # Shut down the spark sessions and context
    sc.stop()
