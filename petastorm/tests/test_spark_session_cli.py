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


import argparse

import pytest
from pyspark.sql import SparkSession

from petastorm.tools.spark_session_cli import add_configure_spark_arguments, configure_spark


@pytest.fixture(scope='session')
def configured_arg_parser():
    parser = argparse.ArgumentParser()
    add_configure_spark_arguments(parser)
    return parser


def test_default_values(configured_arg_parser):
    args = configured_arg_parser.parse_args([])
    assert args.master is None
    assert not args.spark_session_config


def test_some_values(configured_arg_parser):
    args = configured_arg_parser.parse_args(['--master', 'local', '--spark-session-config', 'a=1', 'b=2'])
    assert args.master == 'local'
    assert args.spark_session_config == ['a=1', 'b=2']


def test_session_config(configured_arg_parser):
    args = configured_arg_parser.parse_args(['--master', 'local[1]', '--spark-session-config', 'a=1', 'b=2'])
    spark = configure_spark(SparkSession.builder, args).getOrCreate()
    assert spark.conf.get('a') == '1'
    assert spark.conf.get('b') == '2'
    assert spark.conf.get('spark.master') == 'local[1]'


def test_unconfigured_argparser():
    args = argparse.ArgumentParser().parse_args([])
    with pytest.raises(RuntimeError, match='add_configure_spark_arguments'):
        configure_spark(SparkSession.builder, args)


def test_invalid_key_value_setting(configured_arg_parser):
    args = configured_arg_parser.parse_args(['--spark-session-config', 'WRONG FORMAT', 'b=2', '--master', 'local[1]'])
    with pytest.raises(ValueError, match='key=value'):
        configure_spark(SparkSession.builder, args)
