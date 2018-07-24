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
from pyspark.sql import SparkSession

from petastorm.etl.dataset_metadata import add_dataset_metadata

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
                        help='the fully qualified class of the dataset unischema '
                             '(e.g. av.perception.datasets.msds.schema.MsdsSchema)', required=True)
    parser.add_argument('--master', type=str,
                        help='Spark master. Default if not specified. To run on a local machine, specify '
                             '"local[W]" (where W is the number of local spark workers, e.g. local[10])')
    parser.add_argument('--cores', type=int,
                        help='Number of cores that would be used to run this job (fed into spark.cores.max)')
    args = parser.parse_args()

    dataset_url = args.dataset_url
    schema = locate(args.unischema_class)

    if not schema:
        raise ValueError('Schema {} could not be located'.format(args.unischema_class))

    # Open Spark Session
    spark_session = SparkSession \
        .builder \
        .appName("Petastorm Metadata Index")
    if args.master:
        spark_session.master(args.master)
    if args.cores:
        spark_session.config("spark.cores.max", str(args.cores))

    spark = spark_session.getOrCreate()
    # Spark Context is available from the SparkSession
    sc = spark.sparkContext

    add_dataset_metadata(dataset_url, sc, schema)

    # Shut down the spark sessions and context
    sc.stop()
