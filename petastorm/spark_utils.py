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

"""A set of Spark specific helper functions for the petastorm dataset"""
from six.moves.urllib.parse import urlparse

from petastorm import utils
from petastorm.etl.dataset_metadata import get_schema_from_dataset_url
from petastorm.fs_utils import FilesystemResolver


def dataset_as_rdd(dataset_url, spark_session, schema_fields=None):
    """
    Retrieve a spark rdd for a given petastorm dataset

    :param dataset_url: A string for the dataset url (e.g. hdfs:///path/to/dataset)
    :param spark_session: A spark session
    :param schema_fields: list of unischema fields to subset, or None to read all fields.
    :return: A rdd of dictionary records from the dataset
    """
    schema = get_schema_from_dataset_url(dataset_url)

    dataset_url_parsed = urlparse(dataset_url)

    resolver = FilesystemResolver(dataset_url_parsed, spark_session.sparkContext._jsc.hadoopConfiguration())

    dataset_df = spark_session.read.parquet(resolver.parsed_dataset_url().path)
    if schema_fields is not None:
        # If wanting a subset of fields, create the schema view and run a select on those fields
        schema = schema.create_schema_view(schema_fields)
        field_names = [field.name for field in schema_fields]
        dataset_df = dataset_df.select(*field_names)

    dataset_rows = dataset_df.rdd \
        .map(lambda row: utils.decode_row(row.asDict(), schema)) \
        .map(lambda record: schema.make_namedtuple(**record))

    return dataset_rows
