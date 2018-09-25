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

"""This module contains a set of utils that enables uniform interface for all command line tools that end up creating
spark session objects"""


def configure_spark(spark_session_builder, args):
    """Applies configuration to a ``SparkSession.Builder`` object.

    Call :func:`add_configure_spark_arguments` to add command line arguments to the argparser object.
    This function returns the ``SparkSession.Builder`` to allow chaining additional calls to the ``Builder``.

    >>> from pyspark.sql import SparkSession
    >>>
    >>> arg_parser = argparse.ArgumentParser()
    >>> add_configure_spark_arguments(arg_parser)
    >>> # ... more argparse arguments

    >>> args = arg_parser.parse_args()
    >>> spark = configure_spark(SparkSession.builder.appName('petastorm-copy'), args).getOrCreate()

    :param spark_session_builder: An instance of the ``pyspark.sql.session.SparkSession.Builder`` object.
    :param args: A value returned by ``argparser.ArgumentParser.parse_args()`` call.
    :return: ``SparkSession.Builder`` object.
    """
    if 'spark_session_config' not in args or 'master' not in args:
        raise RuntimeError('--spark-session-config and/or --master were not found in parsed arguments. '
                           'Call add_configure_spark_arguments() to add them.')

    spark_session_config = _cli_spark_session_config_to_dict(args.spark_session_config)

    for key, value in spark_session_config.items():
        spark_session_builder.config(key, value)

    if args.master:
        spark_session_builder.master(args.master)

    return spark_session_builder


def add_configure_spark_arguments(argparser):
    """Adds a set of arguments that are needed for spark session configuration.

    >>> from pyspark.sql import SparkSession
    >>>
    >>> arg_parser = argparse.ArgumentParser()
    >>> add_configure_spark_arguments(arg_parser)
    >>> # ... more argparse arguments

    >>> args = arg_parser.parse_args()
    >>> spark = configure_spark(SparkSession.builder.appName('petastorm-copy'), args).getOrCreate()

    :param argparser: An instance of ``argparse.ArgumentParser`` object
    :return: None
    """
    argparser.add_argument('--master', type=str,
                           help='Spark master. Default if not specified. To run on a local machine, specify '
                                '"local[W]" (where W is the number of local spark workers, e.g. local[10])')

    argparser.add_argument('--spark-session-config', type=str, nargs='+',
                           help='A list of "=" separated key-value pairs used to configure SparkSession object. '
                                'For example: --spark-session-config spark.executor.cores=2 spark.executor.memory=10g')


def _cli_spark_session_config_to_dict(spark_session_config):
    config_dict = dict()

    if not spark_session_config:
        return config_dict

    for config_pair in spark_session_config:
        key_value_split = config_pair.split('=')
        if len(key_value_split) != 2:
            raise ValueError('Elements of spark_session_config list are expected to be in key=value format. Got: %s',
                             config_pair)
        config_dict[key_value_split[0]] = key_value_split[1]

    return config_dict
