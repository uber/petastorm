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

import logging
import time
from collections import namedtuple

from pyarrow import parquet as pq
from six.moves import cPickle as pickle
from six.moves import range

from petastorm import utils
from petastorm.etl import dataset_metadata
from petastorm.etl.legacy import depickle_legacy_package_name_compatible
from petastorm.fs_utils import FilesystemResolver

logger = logging.getLogger(__name__)

PARALLEL_SLICE_NUM = 2000

ROWGROUPS_INDEX_KEY = b'dataset-toolkit.rowgroups_index.v1'

PieceInfo = namedtuple('PieceInfo', ['piece_index', 'path', 'row_group', 'partition_keys'])


def build_rowgroup_index(dataset_url, spark_context, indexers, hdfs_driver='libhdfs3'):
    """
    Build index for given list of fields to use for fast rowgroup selection
    :param dataset_url: (str) the url for the dataset (or a path if you would like to use the default hdfs config)
    :param spark_context: (SparkContext)
    :param indexers: list of objects to build row groups indexes. Should support RowGroupIndexerBase interface
    :param hdfs_driver: A string denoting the hdfs driver to use (if using a dataset on hdfs). Current choices are
    libhdfs (java through JNI) or libhdfs3 (C++)
    :return: None, upon successful completion the rowgroup predicates will be saved to _metadata file
    """

    if dataset_url and dataset_url[-1] == '/':
        dataset_url = dataset_url[:-1]

    # Create pyarrow file system
    resolver = FilesystemResolver(dataset_url, spark_context._jsc.hadoopConfiguration(), hdfs_driver=hdfs_driver)
    dataset = pq.ParquetDataset(resolver.get_dataset_path(), filesystem=resolver.filesystem(),
                                validate_schema=False)

    split_pieces = dataset_metadata.load_row_groups(dataset)
    schema = dataset_metadata.get_schema(dataset)

    # We need direct reference on partitions object
    partitions = dataset.partitions
    pieces_num = len(split_pieces)
    piece_info_list = []
    for piece_index in range(pieces_num):
        #  indexes relies on the ordering of the split dataset pieces.
        # This relies on how the dataset pieces are split and sorted which although should not change,
        # still might and we should make sure not to forget that could break this.
        piece = split_pieces[piece_index]
        piece_info_list.append(PieceInfo(piece_index, piece.path, piece.row_group, piece.partition_keys))

    start_time = time.time()
    piece_info_rdd = spark_context.parallelize(piece_info_list, min(len(piece_info_list), PARALLEL_SLICE_NUM))
    indexer_rdd = piece_info_rdd.map(lambda piece_info: _index_columns(piece_info, dataset_url, partitions,
                                                                       indexers, schema, hdfs_driver=hdfs_driver))
    indexer_list = indexer_rdd.reduce(_combine_indexers)

    indexer_dict = {indexer.index_name: indexer for indexer in indexer_list}
    serialized_indexers = pickle.dumps(indexer_dict, pickle.HIGHEST_PROTOCOL)
    utils.add_to_dataset_metadata(dataset, ROWGROUPS_INDEX_KEY, serialized_indexers)
    logger.info("Elapsed time of index creation: %f s", (time.time() - start_time))


def _index_columns(piece_info, dataset_url, partitions, indexers, schema, hdfs_driver='libhdfs3'):
    """
    Function build indexes for  dataset piece described in piece_info
    :param piece_info: description of dataset piece
    :param dataset_url: dataset location
    :param partitions: dataset partitions
    :param indexers: list of indexer objects
    :param schema: dataset schema
    :param hdfs_driver: A string denoting the hdfs driver to use (if using a dataset on hdfs). Current choices are
        libhdfs (java through JNI) or libhdfs3 (C++)
    :return: list of indexers containing index data
    """
    # Create pyarrow piece
    piece = pq.ParquetDatasetPiece(piece_info.path, row_group=piece_info.row_group,
                                   partition_keys=piece_info.partition_keys)

    # Collect column names needed for indexing
    column_names = set()
    for indexer in indexers:
        column_names.update(indexer.column_names)

    # Read columns needed for indexing
    # Resolver in executor context will get hadoop config from environment
    resolver = FilesystemResolver(dataset_url, hdfs_driver=hdfs_driver)
    column_rows = piece.read(
        open_file_func=resolver.filesystem().open,
        columns=list(column_names),
        partitions=partitions).to_pandas().to_dict('records')

    # Decode columns values
    decoded_rows = [utils.decode_row(row, schema) for row in column_rows]
    if not decoded_rows:
        raise ValueError('Cannot build index with empty decoded_rows, columns: {}, partitions: {}'
                         .format(column_names, partitions))

    # Index columns values
    for indexer in indexers:
        indexer.build_index(decoded_rows, piece_info.piece_index)

    # Indexer objects contain index data, it will be consolidated on reduce phace
    return indexers


def _combine_indexers(indexers1, indexers2):
    """ Conbine index data from two indexers
    :param indexers1: list of indexers to combine index data
    :param indexers2: second list of indexers to combine index data
    :return: first list of indexers containing index data from both indexers in pair"""
    if len(indexers1) != len(indexers2):
        raise ValueError('Cannot reduce results with different dimensions')

    return [indexer_pair[0] + indexer_pair[1] for indexer_pair in zip(indexers1, indexers2)]


def get_row_group_indexes(dataset):
    """
    Extract and return row group indexes from dataset
    :param dataset: dataset object
    :return: dataset indexes as dictionary
    """
    if not dataset.common_metadata:
        raise ValueError('Could not find _metadata file. add_dataset_metadata(..) in'
                         ' petastorm.etl.dataset_metadata.py should be used to'
                         ' generate this file in your ETL code.'
                         ' You can generate it on an existing dataset using rowgroup_indexing_run.py')

    dataset_metadata_dict = dataset.common_metadata.metadata

    # Load rowgroups_index
    if ROWGROUPS_INDEX_KEY not in dataset_metadata_dict:
        raise ValueError('Row groups index is not available in the dataset metadata file. '
                         'You can generate it on an existing dataset using rowgroup_indexing_run.py')

    serialized_indexes = dataset_metadata_dict[ROWGROUPS_INDEX_KEY]

    index_dict = depickle_legacy_package_name_compatible(serialized_indexes)
    return index_dict
