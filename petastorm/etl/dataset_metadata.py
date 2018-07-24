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

import cPickle as pickle
import json
import os
import sys
from contextlib import contextmanager
from operator import attrgetter

from pyarrow import parquet as pq

from petastorm import utils
from petastorm.fs_utils import FilesystemResolver

ROW_GROUPS_PER_FILE_KEY = 'dataset-toolkit.num_row_groups_per_file.v1'
ROW_GROUPS_PER_FILE_KEY_ABSOLUTE_PATHS = 'dataset-toolkit.num_row_groups_per_file'
UNISCHEMA_KEY = 'dataset-toolkit.unischema.v1'


@contextmanager
def materialize_dataset(spark, dataset_url, schema, row_group_size_mb=None):
    """
    A Context Manager which handles all the initialization and finalization necessary
    to generate metadata for a petastorm dataset. This should be used around your
    spark logic to materialize a dataset (specifically the writing of parquet output).

    Note: Any rowgroup indexing should happen outside the materialize_dataset block

    e.g.
    spark = SparkSession.builder...
    dataset_url = 'hdfs:///path/to/my/dataset'
    with materialize_dataset(spark, dataset_url, MyUnischema, 64):
      spark.sparkContext.parallelize(range(0, 10)).\
        ...
        .write.parquet(dataset_url)

    indexers = [SingleFieldIndexer(...)]
    build_rowgroup_index(dataset_url, spark.sparkContext, indexers)

    :param spark The spark session you are using
    :param dataset_url The dataset url to output your dataset to (e.g. hdfs:///path/to/dataset)
    :param schema The unischema definition of your dataset
    :param row_group_size_mb The parquet row group size to use for your dataset
    """
    spark_config = {}
    _init_spark(spark, spark_config, row_group_size_mb)
    yield
    add_dataset_metadata(dataset_url, spark.sparkContext, schema)
    _cleanup_spark(spark, spark_config, row_group_size_mb)


def _init_spark(spark, current_spark_config, row_group_size_mb=None):
    """
    Initializes spark and hdfs config with necessary options for petastorm datasets
    before running the spark job.
    """
    hadoop_config = spark.sparkContext._jsc.hadoopConfiguration()

    # Store current values so we can restore them later
    current_spark_config['parquet.enable.summary-metadata'] = \
        hadoop_config.get('parquet.enable.summary-metadata')
    current_spark_config['parquet.block.size.row.check.min'] = \
        hadoop_config.get('parquet.block.size.row.check.min')
    current_spark_config['parquet.row-group.size.row.check.min'] = \
        hadoop_config.get('parquet.row-group.size.row.check.min')
    current_spark_config['parquet.block.size'] = \
        hadoop_config.get('parquet.block.size')

    hadoop_config.setBoolean('parquet.enable.summary-metadata', False)
    # In our atg fork this config is called parquet.block.size.row.check.min however in newer
    # parquet versions it will be renamed to parquet.row-group.size.row.check.min
    # We use both for backwards compatibility
    # Setting 'parquet.block.size.row.check.min' to 1 results in invalid parquet file
    hadoop_config.setInt('parquet.block.size.row.check.min', 3)
    hadoop_config.setInt('parquet.row-group.size.row.check.min', 3)
    if row_group_size_mb:
        hadoop_config.setInt('parquet.block.size', row_group_size_mb * 1024 * 1024)


def _cleanup_spark(spark, current_spark_config, row_group_size_mb=None):
    """
    Cleans up config changes performed in _init_spark
    """
    hadoop_config = spark.sparkContext._jsc.hadoopConfiguration()

    for key, val in current_spark_config.items():
        if val is not None:
            hadoop_config.set(key, val)
        else:
            hadoop_config.unset(key)


def add_dataset_metadata(dataset_url, spark_context, schema):
    """
    Adds all the metadata to the dataset needed to read the data using petastorm
    :param dataset_url: (str) the url for the dataset (or a path if you would like to use the default hdfs config)
    :param spark_context: (SparkContext)
    :param schema: (Unischema) the schema for the dataset
    :return: None, upon successful completion the metadata file will exist
    """
    resolver = FilesystemResolver(dataset_url, spark_context._jsc.hadoopConfiguration())
    dataset = pq.ParquetDataset(
        resolver.parsed_dataset_url().path,
        filesystem=resolver.filesystem(),
        validate_schema=False)

    _generate_num_row_groups_per_file_metadata(dataset, spark_context)
    _generate_unischema_metadata(dataset, schema)


def _generate_num_row_groups_per_file_metadata(dataset, spark_context):
    """
    Generates the metadata file containing the number of row groups in each file
    for the parquet dataset located at the dataset_url. It does this in spark by
    opening all parquet files in the dataset on the executors and collecting the
    number of row groups in each file back on the driver.

    :param dataset_url: string url for the parquet dataset. Needs to be a directory.
    :param spark_context: spark context to use for retrieving the number of row groups
    in each parquet file in parallel
    :return: None, upon successful completion the metadata file will exist.
    """
    if not isinstance(dataset.paths, str):
        raise ValueError('Expected dataset.paths to be a single path, not a list of paths')

    # Get the common prefix of all the base path in order to retrieve a relative path
    paths = [piece.path for piece in dataset.pieces]

    # Needed pieces from the dataset must be extracted for spark because the dataset object is not serializable
    fs = dataset.fs
    base_path = dataset.paths
    row_groups = spark_context.parallelize(paths, len(paths)) \
        .map(lambda path: (os.path.relpath(path, base_path), pq.read_metadata(fs.open(path)).num_row_groups)) \
        .collect()
    num_row_groups_str = json.dumps(dict(row_groups))
    # Add the dict for the number of row groups in each file to the parquet file metadata footer
    utils.add_to_dataset_metadata(dataset, ROW_GROUPS_PER_FILE_KEY, num_row_groups_str)


def _generate_unischema_metadata(dataset, schema):
    """
    Generates the serialized unischema and adds it to the dataset parquet metadata to be used upon reading.
    :param dataset: (ParquetDataset) Dataset to attach schema
    :param schema:  (Unischema) Schema to attach to dataset
    :return: None
    """
    # TODO(robbieg): Simply pickling unischema will break if the UnischemaField class is changed,
    #  or the codec classes are changed. We likely need something more robust.
    serialized_schema = pickle.dumps(schema)
    utils.add_to_dataset_metadata(dataset, UNISCHEMA_KEY, serialized_schema)


def load_rowgroup_split(dataset):
    """
    Load dataset row group pieces from metadata
    :param dataset: parquet dataset object.
    :return: splitted pieces, one piece per row group
    """
    # Split the dataset pieces by row group using the precomputed index
    if not dataset.common_metadata:
        raise ValueError('Could not find _metadata file. add_dataset_metadata(..) in'
                         ' petastorm.etl.dataset_metadata.py should be used to'
                         ' generate this file in your ETL code.'
                         ' You can generate it on an existing dataset using metadata_index_run.py')

    dataset_metadata_dict = dataset.common_metadata.metadata

    use_absolute_paths = False
    if ROW_GROUPS_PER_FILE_KEY not in dataset_metadata_dict:
        # We also need to check for using absolute paths for backwards compatibility with older generated metadata
        use_absolute_paths = True
        if ROW_GROUPS_PER_FILE_KEY_ABSOLUTE_PATHS not in dataset_metadata_dict:
            raise ValueError('Could not find the row groups per file in the dataset metadata file.'
                             ' Metadata file might not be generated properly.'
                             ' Make sure to use add_dataset_metadata(..) in'
                             ' petastorm.etl.dataset_metadata.py to'
                             ' properly generate this file in your ETL code.'
                             ' You can generate it on an existing dataset using metadata_index_run.py')
    if use_absolute_paths:
        metadata_dict_key = ROW_GROUPS_PER_FILE_KEY_ABSOLUTE_PATHS
    else:
        metadata_dict_key = ROW_GROUPS_PER_FILE_KEY
    row_groups_per_file = json.loads(dataset_metadata_dict[metadata_dict_key])

    split_pieces = []
    # Force order of pieces. The order is not deterministic since it depends on multithreaded directory
    # listing implementation inside pyarrow. We stabilize order here, this way we get reproducable order
    # when pieces shuffling is off. This also enables implementing piece shuffling given a seed
    sorted_pieces = sorted(dataset.pieces, key=attrgetter('path'))
    for piece in sorted_pieces:
        # If we are not using absolute paths, we need to convert the path to a relative path for
        # looking up the number of row groups.
        row_groups_key = piece.path if use_absolute_paths else os.path.relpath(piece.path, dataset.paths)
        for row_group in range(row_groups_per_file[row_groups_key]):
            split_pieces.append(pq.ParquetDatasetPiece(piece.path, row_group, piece.partition_keys))
    return split_pieces


def get_schema(dataset):
    """
    Retrieve schema object stored as part of dataset methadata
    :param dataset:
    :return: unischema object
    """
    # Split the dataset pieces by row group using the precomputed index
    if not dataset.common_metadata:
        raise ValueError('Could not find _metadata file. add_dataset_metadata(..) in'
                         ' petastorm.etl.dataset_metadata.py should be used to'
                         ' generate this file in your ETL code.'
                         ' You can generate it on an existing dataset using metadata_index_run.py')

    dataset_metadata_dict = dataset.common_metadata.metadata

    # Read schema
    if UNISCHEMA_KEY not in dataset_metadata_dict:
        raise ValueError('Could not find the unischema in the dataset metadata file.'
                         ' Please provide or generate dataset with the unischema attached.'
                         ' Metadata file might not be generated properly.'
                         ' Make sure to use add_dataset_metadata(..) in'
                         ' petastorm.etl.dataset_metadata.py to'
                         ' properly generate this file in your ETL code.'
                         ' You can generate it on an existing dataset using metadata_index_run.py')
    ser_schema = dataset_metadata_dict[UNISCHEMA_KEY]
    # Since we have moved the unischema class from av.experimental.deepdrive.dataset_toolkit to dataset_toolkit
    # unpickling old schemas will not work. In this case we override the old import path to get backwards compatibility
    try:
        schema = pickle.loads(ser_schema)
    except ImportError:
        import petastorm
        sys.modules['av.experimental.deepdrive.dataset_toolkit'] = petastorm
        schema = pickle.loads(ser_schema)
    return schema
