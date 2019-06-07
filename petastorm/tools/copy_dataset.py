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


"""This command line utility creates a copy of a Petastorm dataset while optionally:
 - selects a set of columns
 - filters out rows with null values in specified fields."""

import argparse
import logging
import operator
import sys
from functools import reduce  # pylint: disable=W0622

from pyspark.sql import SparkSession

from petastorm.unischema import match_unischema_fields
from petastorm.etl.dataset_metadata import materialize_dataset, get_schema_from_dataset_url
from petastorm.tools.spark_session_cli import add_configure_spark_arguments, configure_spark
from petastorm.fs_utils import FilesystemResolver


def copy_dataset(spark, source_url, target_url, field_regex, not_null_fields, overwrite_output, partitions_count,
                 row_group_size_mb, hdfs_driver='libhdfs3'):
    """
    Creates a copy of a dataset. A new dataset will optionally contain a subset of columns. Rows that have NULL
    values in fields defined by ``not_null_fields`` argument are filtered out.


    :param spark: An instance of ``SparkSession`` object
    :param source_url: A url of the dataset to be copied.
    :param target_url: A url specifying location of the target dataset.
    :param field_regex: A list of regex patterns. Only columns that match one of these patterns are copied to the new
      dataset.
    :param not_null_fields: A list of fields that must have non-NULL valus in the target dataset.
    :param overwrite_output: If ``False`` and there is an existing path defined by ``target_url``, the operation will
      fail.
    :param partitions_count: If not ``None``, the dataset is repartitioned before write. Number of files in the target
      Parquet store is defined by this parameter.
    :param row_group_size_mb: The size of the rowgroup in the target dataset. Specified in megabytes.
    :param hdfs_driver: A string denoting the hdfs driver to use (if using a dataset on hdfs). Current choices are
        libhdfs (java through JNI) or libhdfs3 (C++)
    :param user: String denoting username when connecting to HDFS. None implies login user.
    :return: None
    """
    schema = get_schema_from_dataset_url(source_url, hdfs_driver=hdfs_driver)

    fields = match_unischema_fields(schema, field_regex)

    if field_regex and not fields:
        field_names = list(schema.fields.keys())
        raise ValueError('Regular expressions (%s) do not match any fields (%s)', str(field_regex), str(field_names))

    if fields:
        subschema = schema.create_schema_view(fields)
    else:
        subschema = schema

    resolver = FilesystemResolver(target_url, spark.sparkContext._jsc.hadoopConfiguration(),
                                  hdfs_driver=hdfs_driver, user=spark.sparkContext.sparkUser())
    with materialize_dataset(spark, target_url, subschema, row_group_size_mb,
                             filesystem_factory=resolver.filesystem_factory()):
        data_frame = spark.read \
            .parquet(source_url)

        if fields:
            data_frame = data_frame.select(*[f.name for f in fields])

        if not_null_fields:
            not_null_condition = reduce(operator.__and__, (data_frame[f].isNotNull() for f in not_null_fields))
            data_frame = data_frame.filter(not_null_condition)

        if partitions_count:
            data_frame = data_frame.repartition(partitions_count)

        data_frame.write \
            .mode('overwrite' if overwrite_output else 'error') \
            .option('compression', 'none') \
            .parquet(target_url)


def args_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('source_url',
                        help='A url of a source petastorm dataset',
                        type=str)

    parser.add_argument('target_url',
                        help='A url of a target petastorm datset',
                        type=str)

    parser.add_argument('--overwrite-output', action='store_true',
                        help='If the flag is set to false, the script will fail '
                             'in case when the output directory already exists')

    parser.add_argument('--field-regex', type=str, nargs='+',
                        help='A list of regular expressions. Only fields that match one of the regex patterns will '
                             'be copied.')

    parser.add_argument('--not-null-fields', type=str, nargs='+',
                        help='All names in this list must be not null in the source dataset in order to be copied to '
                             'the target dataset.')

    parser.add_argument('--partition-count', type=int, required=False,
                        help='Specifies number of partitions in the output dataset')

    parser.add_argument('--row-group-size-mb', type=int, required=False,
                        help='Specifies the row group size in the created dataset')
    parser.add_argument('--hdfs-driver', type=str, default='libhdfs3',
                        help='A string denoting the hdfs driver to use (if using a dataset on hdfs). '
                             'Current choices are libhdfs (java through JNI) or libhdfs3 (C++)')

    add_configure_spark_arguments(parser)

    return parser


def _main(sys_argv):
    logging.basicConfig()

    args = args_parser().parse_args(sys_argv)

    # We set spark.sql.files.maxPartitionBytes to a large value since we typically have small number of rows per
    # rowgroup. Reading a parquet store with default settings would result in excessively large number of partitions
    # and inefficient processing
    spark = configure_spark(SparkSession.builder.appName('petastorm-copy'), args) \
        .config('spark.sql.files.maxPartitionBytes', '1010612736') \
        .getOrCreate()

    copy_dataset(spark, args.source_url, args.target_url, args.field_regex, args.not_null_fields, args.overwrite_output,
                 args.partition_count, args.row_group_size_mb, hdfs_driver=args.hdfs_driver)

    spark.stop()


def main():
    _main(sys.argv[1:])


if __name__ == '__main__':
    _main(sys.argv[1:])
