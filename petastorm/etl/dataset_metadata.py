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
from concurrent import futures
from contextlib import contextmanager
from operator import attrgetter

from pyarrow import parquet as pq
from six.moves import cPickle as pickle
from six.moves.urllib.parse import urlparse

from petastorm import utils
from petastorm.etl.legacy import depickle_legacy_package_name_compatible
from petastorm.fs_utils import FilesystemResolver
from petastorm.unischema import Unischema

logger = logging.getLogger(__name__)

ROW_GROUPS_PER_FILE_KEY = b'dataset-toolkit.num_row_groups_per_file.v1'
UNISCHEMA_KEY = b'dataset-toolkit.unischema.v1'


class PetastormMetadataError(Exception):
    """
    Error to specify when the petastorm metadata does not exist, does not contain the necessary information,
    or is corrupt/invalid.
    """


class PetastormMetadataGenerationError(Exception):
    """
    Error to specify when petastorm could not generate metadata properly.
    This error is usually accompanied with a message to try to regenerate dataset metadata.
    """


@contextmanager
def materialize_dataset(spark, dataset_url, schema, row_group_size_mb=None, use_summary_metadata=False,
                        filesystem_factory=None):
    """
    A Context Manager which handles all the initialization and finalization necessary
    to generate metadata for a petastorm dataset. This should be used around your
    spark logic to materialize a dataset (specifically the writing of parquet output).

    Note: Any rowgroup indexing should happen outside the materialize_dataset block

    Example:

    >>> spark = SparkSession.builder...
    >>> ds_url = 'hdfs:///path/to/my/dataset'
    >>> with materialize_dataset(spark, ds_url, MyUnischema, 64):
    >>>   spark.sparkContext.parallelize(range(0, 10)).
    >>>     ...
    >>>     .write.parquet(ds_url)
    >>> indexer = [SingleFieldIndexer(...)]
    >>> build_rowgroup_index(ds_url, spark.sparkContext, indexer)

    A user may provide their own instance of pyarrow filesystem object in ``pyarrow_filesystem`` argument (otherwise,
    petastorm will create a default one based on the url).

    The following example shows how a custom pyarrow HDFS filesystem, instantiated using ``libhdfs`` driver can be used
    during Petastorm dataset generation:

    >>> resolver=FilesystemResolver(dataset_url, spark.sparkContext._jsc.hadoopConfiguration(),
    >>>                             hdfs_driver='libhdfs')
    >>> with materialize_dataset(..., pyarrow_filesystem=resolver.filesystem()):
    >>>     ...


    :param spark: The spark session you are using
    :param dataset_url: The dataset url to output your dataset to (e.g. ``hdfs:///path/to/dataset``)
    :param schema: The :class:`petastorm.unischema.Unischema` definition of your dataset
    :param row_group_size_mb: The parquet row group size to use for your dataset
    :param use_summary_metadata: Whether to use the parquet summary metadata for row group indexing or a custom
      indexing method. The custom indexing method is more scalable for very large datasets.
    :param pyarrow_filesystem: A pyarrow filesystem object to be used when saving Petastorm specific metadata to the
      Parquet store.

    """
    spark_config = {}
    _init_spark(spark, spark_config, row_group_size_mb, use_summary_metadata)
    yield

    # After job completes, add the unischema metadata and check for the metadata summary file
    if filesystem_factory is None:
        resolver = FilesystemResolver(dataset_url, spark.sparkContext._jsc.hadoopConfiguration())
        filesystem_factory = resolver.filesystem_factory()
        dataset_path = resolver.get_dataset_path()
    else:
        dataset_path = urlparse(dataset_url).path
    filesystem = filesystem_factory()

    dataset = pq.ParquetDataset(
        dataset_path,
        filesystem=filesystem,
        validate_schema=False)

    _generate_unischema_metadata(dataset, schema)
    if not use_summary_metadata:
        _generate_num_row_groups_per_file(dataset, spark.sparkContext, filesystem_factory)

    # Reload the dataset to take into account the new metadata
    dataset = pq.ParquetDataset(
        dataset_path,
        filesystem=filesystem,
        validate_schema=False)
    try:
        # Try to load the row groups, if it fails that means the metadata was not generated properly
        load_row_groups(dataset)
    except PetastormMetadataError:
        raise PetastormMetadataGenerationError(
            'Could not find summary metadata file. The dataset will exist but you will need'
            ' to execute petastorm-generate-metadata.py before you can read your dataset '
            ' in order to generate the necessary metadata.'
            ' Try increasing spark driver memory next time and making sure you are'
            ' using parquet-mr >= 1.8.3')

    _cleanup_spark(spark, spark_config, row_group_size_mb)


def _init_spark(spark, current_spark_config, row_group_size_mb=None, use_summary_metadata=False):
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

    hadoop_config.setBoolean('parquet.enable.summary-metadata', use_summary_metadata)
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
    assert schema
    serialized_schema = pickle.dumps(schema)
    utils.add_to_dataset_metadata(dataset, UNISCHEMA_KEY, serialized_schema)


def _generate_num_row_groups_per_file(dataset, spark_context, filesystem_factory):
    """
    Generates the metadata file containing the number of row groups in each file
    for the parquet dataset located at the dataset_url. It does this in spark by
    opening all parquet files in the dataset on the executors and collecting the
    number of row groups in each file back on the driver.
    :param dataset: :class:`pyarrow.parquet.ParquetDataset`
    :param spark_context: spark context to use for retrieving the number of row groups
    in each parquet file in parallel
    :return: None, upon successful completion the metadata file will exist.
    """
    if not isinstance(dataset.paths, str):
        raise ValueError('Expected dataset.paths to be a single path, not a list of paths')

    # Get the common prefix of all the base path in order to retrieve a relative path
    paths = [piece.path for piece in dataset.pieces]

    # Needed pieces from the dataset must be extracted for spark because the dataset object is not serializable
    base_path = dataset.paths

    def get_row_group_info(path):
        fs = filesystem_factory()
        relative_path = os.path.relpath(path, base_path)
        pq_file = fs.open(path)
        num_row_groups = pq.read_metadata(pq_file).num_row_groups
        pq_file.close()
        return relative_path, num_row_groups

    row_groups = spark_context.parallelize(paths, len(paths)) \
        .map(get_row_group_info) \
        .collect()
    num_row_groups_str = json.dumps(dict(row_groups))
    # Add the dict for the number of row groups in each file to the parquet file metadata footer
    utils.add_to_dataset_metadata(dataset, ROW_GROUPS_PER_FILE_KEY, num_row_groups_str)


def load_row_groups(dataset):
    """
    Load dataset row group pieces from metadata
    :param dataset: parquet dataset object.
    :param allow_read_footers: whether to allow reading parquet footers if there is no better way
            to load row group information
    :return: splitted pieces, one piece per row group
    """
    # We try to get row group information from metadata file
    metadata = dataset.metadata
    common_metadata = dataset.common_metadata
    if not metadata and not common_metadata:
        # If we are inferring the schema we allow reading the footers to get the row group information
        return _split_row_groups_from_footers(dataset)

    if metadata and metadata.num_row_groups > 0:
        # If the metadata file exists and has row group information we use it to split the dataset pieces
        return _split_row_groups(dataset)

    # If we don't have row groups in the common metadata we look for the old way of loading it
    dataset_metadata_dict = common_metadata.metadata
    if ROW_GROUPS_PER_FILE_KEY not in dataset_metadata_dict:
        raise PetastormMetadataError(
            'Could not find row group metadata in _common_metadata file.'
            ' Use materialize_dataset(..) in petastorm.etl.dataset_metadata.py to generate'
            ' this file in your ETL code.'
            ' You can generate it on an existing dataset using petastorm-generate-metadata.py')
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
            rowgroups.append(pq.ParquetDatasetPiece(piece.path, row_group=row_group,
                                                    partition_keys=piece.partition_keys))
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
            split_piece = pq.ParquetDatasetPiece(piece.path, row_group=row_group, partition_keys=piece.partition_keys)
            split_pieces.append(split_piece)

    return split_pieces


def _split_row_groups_from_footers(dataset):
    """Split the row groups by reading the footers of the parquet pieces"""

    logger.info('Recovering rowgroup information for the entire dataset. This can take a long time for datasets with '
                'large number of files. If this dataset was generated by Petastorm '
                '(i.e. by using "with materialize_dataset(...)") and you still see this message, '
                'this indicates that the materialization did not finish successfully.')

    thread_pool = futures.ThreadPoolExecutor()

    def split_piece(piece):
        metadata = piece.get_metadata(dataset.fs.open)
        return [pq.ParquetDatasetPiece(piece.path,
                                       row_group=row_group,
                                       partition_keys=piece.partition_keys)
                for row_group in range(metadata.num_row_groups)]

    futures_list = [thread_pool.submit(split_piece, piece) for piece in dataset.pieces]
    result = [item for f in futures_list for item in f.result()]
    thread_pool.shutdown()
    return result


def get_schema(dataset):
    """Retrieves schema object stored as part of dataset methadata.

    :param dataset: an instance of :class:`pyarrow.parquet.ParquetDataset object`
    :return: A :class:`petastorm.unischema.Unischema` object
    """
    if not dataset.common_metadata:
        raise PetastormMetadataError(
            'Could not find _common_metadata file. Use materialize_dataset(..) in'
            ' petastorm.etl.dataset_metadata.py to generate this file in your ETL code.'
            ' You can generate it on an existing dataset using petastorm-generate-metadata.py')

    dataset_metadata_dict = dataset.common_metadata.metadata

    # Read schema
    if UNISCHEMA_KEY not in dataset_metadata_dict:
        raise PetastormMetadataError(
            'Could not find the unischema in the dataset common metadata file.'
            ' Please provide or generate dataset with the unischema attached.'
            ' Common Metadata file might not be generated properly.'
            ' Make sure to use materialize_dataset(..) in petastorm.etl.dataset_metadata to'
            ' properly generate this file in your ETL code.'
            ' You can generate it on an existing dataset using petastorm-generate-metadata.py')
    ser_schema = dataset_metadata_dict[UNISCHEMA_KEY]
    # Since we have moved the unischema class around few times, unpickling old schemas will not work. In this case we
    # override the old import path to get backwards compatibility

    schema = depickle_legacy_package_name_compatible(ser_schema)

    return schema


def get_schema_from_dataset_url(dataset_url, hdfs_driver='libhdfs3'):
    """Returns a :class:`petastorm.unischema.Unischema` object loaded from a dataset specified by a url.

    :param dataset_url: A dataset URL
    :param hdfs_driver: A string denoting the hdfs driver to use (if using a dataset on hdfs). Current choices are
        libhdfs (java through JNI) or libhdfs3 (C++)
    :return: A :class:`petastorm.unischema.Unischema` object
    """
    resolver = FilesystemResolver(dataset_url, hdfs_driver=hdfs_driver)
    dataset = pq.ParquetDataset(resolver.get_dataset_path(), filesystem=resolver.filesystem(),
                                validate_schema=False)

    # Get a unischema stored in the dataset metadata.
    stored_schema = get_schema(dataset)

    return stored_schema


def infer_or_load_unischema(dataset):
    """Try to recover Unischema object stored by ``materialize_dataset`` function. If it can be loaded, infer
    Unischema from native Parquet schema"""
    try:
        return get_schema(dataset)
    except PetastormMetadataError:
        logger.info('Failed loading Unischema from metadata in %s. Assuming the dataset was not created with '
                    'Petastorm. Will try to construct from native Parquet schema.')
        return Unischema.from_arrow_schema(dataset)
