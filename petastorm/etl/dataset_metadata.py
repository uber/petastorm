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

import json
import logging
import os
import pickle
from contextlib import contextmanager
from operator import attrgetter

from pyarrow import parquet as pq
from six.moves import cPickle as pickle

from petastorm import utils
from petastorm.etl.legacy import depickle_legacy_package_name_compatible
from petastorm.fs_utils import FilesystemResolver


logger = logging.getLogger(__name__)

ROW_GROUPS_PER_FILE_KEY = b'dataset-toolkit.num_row_groups_per_file.v1'
UNISCHEMA_KEY = b'dataset-toolkit.unischema.v1'


class MetadataGenerationError(Exception):
    """
    Error to specify when petastorm could not generate metadata properly.
    This error is usually accompanied with a message to try to regenerate dataset metadata.
    """
    pass


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

    # After job completes, add the unischema metadata and check for the metadata summary file
    resolver = FilesystemResolver(dataset_url, spark.sparkContext._jsc.hadoopConfiguration())
    dataset = pq.ParquetDataset(
        resolver.parsed_dataset_url().path,
        filesystem=resolver.filesystem(),
        validate_schema=False)

    _generate_unischema_metadata(dataset, schema)
    if not dataset.metadata_path:
        raise MetadataGenerationError('Could not find summary metadata file. The dataset will exist but you will need'
                                      ' to execute petastorm-generate-metadata before you can read your dataset '
                                      ' in order to generate the necessary metadata.'
                                      ' Try increasing spark driver memory next time and making sure you are'
                                      ' using parquet-mr >= 1.8.3')

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
    current_spark_config['parquet.summary.metadata.propagate-errors'] = \
        hadoop_config.get('parquet.summary.metadata.propagate-errors')
    current_spark_config['parquet.block.size.row.check.min'] = \
        hadoop_config.get('parquet.block.size.row.check.min')
    current_spark_config['parquet.row-group.size.row.check.min'] = \
        hadoop_config.get('parquet.row-group.size.row.check.min')
    current_spark_config['parquet.block.size'] = \
        hadoop_config.get('parquet.block.size')

    hadoop_config.setBoolean('parquet.enable.summary-metadata', True)
    # Our atg fork includes https://github.com/apache/parquet-mr/pull/502 which creates this
    # option. This forces a job to fail if the summary metadata files cannot be created
    # instead of just having them fail to be created silently
    hadoop_config.setBoolean('parquet.summary.metadata.propagate-errors', True)
    # In our atg fork this config is called parquet.block.size.row.check.min however in newer
    # parquet versions it will be renamed to parquet.row-group.size.row.check.min
    # We use both for backwards compatibility
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


def load_row_groups(dataset):
    """
    Load dataset row group pieces from metadata
    :param dataset: parquet dataset object.
    :return: splitted pieces, one piece per row group
    """
    # Split the dataset pieces by row group
    metadata = dataset.metadata
    if not metadata:
        raise ValueError('Could not find _metadata file.'
                         ' Use materialize_dataset(..) in petastorm.etl.dataset_metadata.py to generate'
                         ' this file in your ETL code.'
                         ' You can generate it on an existing dataset using petastorm-generate-metadata')

    num_row_groups = metadata.num_row_groups

    if num_row_groups > 0:
        # Use the new metadata file
        return _split_row_groups(dataset)

    # If we don't have row groups in the common metadata we look for the old way of loading it
    logger.warning('You are using a deprecated metadata version. Please run petastorm-generate-metadata'
                   ' on spark to update.')
    dataset_metadata_dict = dataset.common_metadata.metadata
    if ROW_GROUPS_PER_FILE_KEY not in dataset_metadata_dict:
        raise ValueError('Could not find row group metadata in _metadata file.'
                         ' Use materialize_dataset(..) in petastorm.etl.dataset_metadata.py to generate'
                         ' this file in your ETL code.'
                         ' You can generate it on an existing dataset using petastorm-generate-metadata')
    metadata_dict_key = ROW_GROUPS_PER_FILE_KEY
    row_groups_per_file = json.loads(dataset_metadata_dict[metadata_dict_key].decode())

    rowgroups = []
    # Force order of pieces. The order is not deterministic since it depends on multithreaded directory
    # listing implementation inside pyarrow. We stabilize order here, this way we get reproducable order
    # when pieces shuffling is off. This also enables implementing piece shuffling given a seed
    sorted_pieces = sorted(dataset.pieces, key=attrgetter('path'))
    for piece in sorted_pieces:
        # If we are not using absolute paths, we need to convert the path to a relative path for
        # looking up the number of row groups.
        row_groups_key = os.path.relpath(piece.path, dataset.paths)
        for row_group in range(row_groups_per_file[row_groups_key]):
            rowgroups.append(pq.ParquetDatasetPiece(piece.path, row_group, piece.partition_keys))
    return rowgroups


# This code has been copied (with small adjustments) from https://github.com/apache/arrow/pull/2223
# Once that is merged and released this code can be deleted since we can use the open source
# implementation.
def _split_row_groups(dataset):
    if not dataset.metadata or dataset.metadata.num_row_groups == 0:
        raise NotImplementedError("split_row_groups is only implemented "
                                  "if dataset has parquet summary files "
                                  "with row group information")

    # We make a dictionary of how many row groups are in each file in
    # order to split them. The Parquet Metadata file stores paths as the
    # relative path from the dataset base dir.
    row_groups_per_file = dict()
    for i in range(dataset.metadata.num_row_groups):
        row_group = dataset.metadata.row_group(i)
        path = row_group.column(0).file_path
        row_groups_per_file[path] = row_groups_per_file.get(path, 0) + 1

    base_path = os.path.normpath(os.path.dirname(dataset.metadata_path))
    split_pieces = []
    for piece in dataset.pieces:
        # Since the pieces are absolute path, we get the
        # relative path to the dataset base dir to fetch the
        # number of row groups in the file
        relative_path = os.path.relpath(piece.path, base_path)

        # If the path is not in the metadata file, that means there are
        # no row groups in that file and that file should be skipped
        if relative_path not in row_groups_per_file:
            continue

        for row_group in range(row_groups_per_file[relative_path]):
            split_piece = pq.ParquetDatasetPiece(piece.path, row_group, piece.partition_keys)
            split_pieces.append(split_piece)

    return split_pieces


def get_schema(dataset):
    """
    Retrieve schema object stored as part of dataset methadata
    :param dataset:
    :return: unischema object
    """
    # Split the dataset pieces by row group using the precomputed index
    if not dataset.common_metadata:
        raise ValueError('Could not find _common_metadata file. Use materialize_dataset(..) in'
                         ' petastorm.etl.dataset_metadata.py to generate this file in your '
                         ' ETL code.'
                         ' You can generate it on an existing dataset using petastorm-generate-metadata')

    dataset_metadata_dict = dataset.common_metadata.metadata

    # Read schema
    if UNISCHEMA_KEY not in dataset_metadata_dict:
        raise ValueError('Could not find the unischema in the dataset common metadata file.'
                         ' Please provide or generate dataset with the unischema attached.'
                         ' Common Metadata file might not be generated properly.'
                         ' Make sure to use materialize_dataset(..) in'
                         ' petastorm.etl.dataset_metadata to'
                         ' properly generate this file in your ETL code.'
                         ' You can generate it on an existing dataset using petastorm-generate-metadata')
    ser_schema = dataset_metadata_dict[UNISCHEMA_KEY]
    # Since we have moved the unischema class around few times, unpickling old schemas will not work. In this case we
    # override the old import path to get backwards compatibility

    schema = depickle_legacy_package_name_compatible(ser_schema)

    return schema
