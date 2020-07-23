#  Copyright (c) 2020 Databricks, Inc.
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

import os
import subprocess
import sys
import tempfile
import threading
import time
from distutils.version import LooseVersion

import numpy as np
import pyspark
import pytest
import py4j
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import (ArrayType, BinaryType, BooleanType, ByteType,
                               DoubleType, FloatType, IntegerType, LongType,
                               ShortType, StringType, StructField, StructType)
from six.moves.urllib.parse import urlparse

from petastorm import make_batch_reader
from petastorm.fs_utils import FilesystemResolver
from petastorm.spark import (SparkDatasetConverter, make_spark_converter,
                             spark_dataset_converter)
from petastorm.spark.spark_dataset_converter import (
    _check_dataset_file_median_size, _check_parent_cache_dir_url,
    _check_rank_and_size_consistent_with_horovod, _check_url,
    _get_horovod_rank_and_size, _get_spark_session, _make_sub_dir_url,
    register_delete_dir_handler, _wait_file_available)

from unittest import mock

from petastorm.tests.test_tf_utils import create_tf_graph


@create_tf_graph
def test_primitive(spark_test_ctx):
    schema = StructType([
        StructField("bool_col", BooleanType(), False),
        StructField("float_col", FloatType(), False),
        StructField("double_col", DoubleType(), False),
        StructField("short_col", ShortType(), False),
        StructField("int_col", IntegerType(), False),
        StructField("long_col", LongType(), False),
        StructField("str_col", StringType(), False),
        StructField("bin_col", BinaryType(), False),
        StructField("byte_col", ByteType(), False),
    ])
    df = spark_test_ctx.spark.createDataFrame(
        [(True, 0.12, 432.1, 5, 5, 0, "hello",
          bytearray(b"spark\x01\x02"), -128),
         (False, 123.45, 0.987, 9, 908, 765, "petastorm",
          bytearray(b"\x0012345"), 127)],
        schema=schema).coalesce(1)
    # If we use numPartition > 1, the order of the loaded dataset would
    # be non-deterministic.
    expected_df = df.collect()

    converter = make_spark_converter(df)
    with converter.make_tf_dataset() as dataset:
        iterator = dataset.make_one_shot_iterator()
        tensor = iterator.get_next()
        with tf.Session() as sess:
            ts = sess.run(tensor)
            # TODO: we will improve the test once the batch_size argument
            #  added.
            # Now we only have one batch.
        for i in range(converter.dataset_size):
            for col in df.schema.names:
                actual_ele = getattr(ts, col)[i]
                expected_ele = expected_df[i][col]
                if col == "str_col":
                    actual_ele = actual_ele.decode()
                if col == "bin_col":
                    actual_ele = bytearray(actual_ele)
                if col == "float_col" or col == "double_col":
                    # Note that the default dtype is float32
                    assert pytest.approx(expected_ele, rel=1e-6) == actual_ele
                else:
                    assert expected_ele == actual_ele

        assert len(expected_df) == len(converter)

    assert np.bool_ == ts.bool_col.dtype.type
    assert np.float32 == ts.float_col.dtype.type
    # Default dtype float32
    assert np.float32 == ts.double_col.dtype.type
    assert np.int16 == ts.short_col.dtype.type
    assert np.int32 == ts.int_col.dtype.type
    assert np.int64 == ts.long_col.dtype.type
    assert np.object_ == ts.str_col.dtype.type
    assert np.object_ == ts.bin_col.dtype.type


@create_tf_graph
def test_array_field(spark_test_ctx):
    @pandas_udf('array<float>')
    def gen_array(v):
        return v.map(lambda x: np.random.rand(10))
    df1 = spark_test_ctx.spark.range(10).withColumn('v', gen_array('id')).repartition(2)
    cv1 = make_spark_converter(df1)
    # we can auto infer one-dim array shape
    with cv1.make_tf_dataset(batch_size=4, num_epochs=1) as dataset:
        tf_iter = dataset.make_one_shot_iterator()
        next_op = tf_iter.get_next()
        with tf.Session() as sess:
            batch1 = sess.run(next_op)
        assert batch1.v.shape == (4, 10)


def test_delete(spark_test_ctx):
    df = spark_test_ctx.spark.createDataFrame([(1, 2), (4, 5)], ["col1", "col2"])
    # TODO add test for hdfs url
    converter = make_spark_converter(df)
    local_path = urlparse(converter.cache_dir_url).path
    assert os.path.exists(local_path)
    converter.delete()
    assert not os.path.exists(local_path)


def test_atexit(spark_test_ctx):
    lines = """
    from petastorm.spark import SparkDatasetConverter, make_spark_converter
    from pyspark.sql import SparkSession
    import os
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, '{temp_url}')
    df = spark.createDataFrame([(1, 2),(4, 5)], ["col1", "col2"])
    converter = make_spark_converter(df)
    f = open(os.path.join('{tempdir}', 'test_atexit.out'), "w")
    f.write(converter.cache_dir_url)
    f.close()
    """.format(tempdir=spark_test_ctx.tempdir, temp_url=spark_test_ctx.temp_url)
    code_str = "; ".join(
        line.strip() for line in lines.strip().splitlines())
    ret_code = subprocess.call([sys.executable, "-c", code_str])
    assert 0 == ret_code
    with open(os.path.join(spark_test_ctx.tempdir, 'test_atexit.out')) as f:
        cache_dir_url = f.read()

    fs = FilesystemResolver(cache_dir_url).filesystem()
    assert not fs.exists(urlparse(cache_dir_url).path)


def test_set_delete_handler(spark_test_ctx):
    def test_delete_handler(dir_url):
        raise RuntimeError('Not implemented delete handler.')

    register_delete_dir_handler(test_delete_handler)

    with pytest.raises(RuntimeError, match='Not implemented delete handler'):
        spark_dataset_converter._delete_dir_handler(spark_test_ctx.temp_url)

    # Restore default delete handler (other test will use it)
    register_delete_dir_handler(None)


def _get_compression_type(data_url):
    files = os.listdir(urlparse(data_url).path)
    pq_files = list(filter(lambda x: x.endswith('.parquet'), files))
    filename_splits = pq_files[0].split('.')
    if len(filename_splits) == 2:
        return "uncompressed"
    else:
        return filename_splits[1]


def test_compression(spark_test_ctx):
    df1 = spark_test_ctx.spark.range(10)

    converter1 = make_spark_converter(df1)
    assert "uncompressed" == \
           _get_compression_type(converter1.cache_dir_url).lower()

    converter2 = make_spark_converter(df1, compression_codec="snappy")
    assert "snappy" == \
           _get_compression_type(converter2.cache_dir_url).lower()


def test_df_caching(spark_test_ctx):
    df1 = spark_test_ctx.spark.range(10)
    df2 = spark_test_ctx.spark.range(10)
    df3 = spark_test_ctx.spark.range(20)

    # Test caching for the dataframes with the same logical plan
    converter1 = make_spark_converter(df1)
    converter2 = make_spark_converter(df2)
    assert converter1.cache_dir_url == converter2.cache_dir_url

    # Test no caching for different dataframes
    converter3 = make_spark_converter(df3)
    assert converter1.cache_dir_url != converter3.cache_dir_url

    # Test no caching for the same dataframe with different row group size
    converter11 = make_spark_converter(
        df1, parquet_row_group_size_bytes=8 * 1024 * 1024)
    converter21 = make_spark_converter(
        df1, parquet_row_group_size_bytes=16 * 1024 * 1024)
    assert converter11.cache_dir_url != converter21.cache_dir_url

    # Test no caching for the same dataframe with different compression_codec
    converter12 = make_spark_converter(df1, compression_codec=None)
    converter22 = make_spark_converter(df1, compression_codec="snappy")
    assert converter12.cache_dir_url != converter22.cache_dir_url

    ori_temp_url = spark_test_ctx.spark.conf.get(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF)
    tempdir = tempfile.mkdtemp('_spark_converter_test1')
    new_temp_url = 'file://' + tempdir.replace(os.sep, '/')
    try:
        # Test no caching for the same dataframe with different parent cache dirs
        spark_test_ctx.spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
                                      new_temp_url)
        assert ori_temp_url != new_temp_url
        converter13 = make_spark_converter(df1)
        assert converter1.cache_dir_url != converter13.cache_dir_url

        # Test caching for the same dataframe with different parent cache dirs
        # that could be normalized to the same parent cache dir
        new_temp_url_2 = new_temp_url + os.sep
        spark_test_ctx.spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
                                      new_temp_url_2)
        assert new_temp_url != new_temp_url_2
        converter14 = make_spark_converter(df1)
        assert converter13.cache_dir_url == converter14.cache_dir_url
    finally:
        spark_test_ctx.spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
                                      ori_temp_url)


def test_df_delete_caching_meta(spark_test_ctx):
    from petastorm.spark.spark_dataset_converter import _cache_df_meta_list
    df1 = spark_test_ctx.spark.range(10)
    df2 = spark_test_ctx.spark.range(20)
    converter1 = make_spark_converter(df1)
    converter2 = make_spark_converter(df2)
    converter1.delete()
    cached_list = set(map(lambda x: x.cache_dir_url, _cache_df_meta_list))
    assert converter1.cache_dir_url not in cached_list
    assert converter2.cache_dir_url in cached_list
    # test recreate converter1 after delete should work.
    make_spark_converter(df1)


def test_check_url():
    with pytest.raises(ValueError, match='scheme-less'):
        _check_url('/a/b/c')


def test_make_sub_dir_url():
    assert _make_sub_dir_url('file:///a/b', 'c') == 'file:///a/b/c'
    assert _make_sub_dir_url('hdfs:/a/b', 'c') == 'hdfs:/a/b/c'
    assert _make_sub_dir_url('hdfs://nn1:9000/a/b', 'c') == 'hdfs://nn1:9000/a/b/c'


def test_pickling_remotely(spark_test_ctx):
    df1 = spark_test_ctx.spark.range(100, 101)
    converter1 = make_spark_converter(df1)

    @create_tf_graph
    def map_fn(_):
        with converter1.make_tf_dataset() as dataset:
            iterator = dataset.make_one_shot_iterator()
            tensor = iterator.get_next()
            with tf.Session() as sess:
                ts = sess.run(tensor)
        return getattr(ts, 'id')[0]

    result = spark_test_ctx.spark.sparkContext.parallelize(range(1), 1).map(map_fn).collect()[0]
    assert result == 100


@create_tf_graph
def test_tf_dataset_batch_size(spark_test_ctx):
    df1 = spark_test_ctx.spark.range(100)

    batch_size = 30
    converter1 = make_spark_converter(df1)

    with converter1.make_tf_dataset(batch_size=batch_size) as dataset:
        iterator = dataset.make_one_shot_iterator()
        tensor = iterator.get_next()
        with tf.Session() as sess:
            ts = sess.run(tensor)
    assert len(ts.id) == batch_size


@mock.patch('petastorm.spark.spark_dataset_converter.make_batch_reader')
def test_tf_dataset_petastorm_args(mock_make_batch_reader, spark_test_ctx):
    df1 = spark_test_ctx.spark.range(100).repartition(4)
    conv1 = make_spark_converter(df1)

    mock_make_batch_reader.return_value = make_batch_reader(conv1.cache_dir_url)

    with conv1.make_tf_dataset(reader_pool_type='dummy', cur_shard=1, shard_count=4):
        pass
    peta_args = mock_make_batch_reader.call_args[1]
    assert peta_args['reader_pool_type'] == 'dummy' and \
        peta_args['cur_shard'] == 1 and \
        peta_args['shard_count'] == 4 and \
        peta_args['num_epochs'] is None and \
        peta_args['workers_count'] == 4

    with conv1.make_tf_dataset(num_epochs=1, workers_count=2):
        pass
    peta_args = mock_make_batch_reader.call_args[1]
    assert peta_args['num_epochs'] == 1 and peta_args['workers_count'] == 2


def test_horovod_rank_compatibility(spark_test_ctx):
    with mock.patch.dict(os.environ, {'HOROVOD_RANK': '1', 'HOROVOD_SIZE': '3'}, clear=True):
        assert (1, 3) == _get_horovod_rank_and_size()
        assert _check_rank_and_size_consistent_with_horovod(
            petastorm_reader_kwargs={"cur_shard": 1, "shard_count": 3})
        assert not _check_rank_and_size_consistent_with_horovod(
            petastorm_reader_kwargs={"cur_shard": 1, "shard_count": 2})
        assert not _check_rank_and_size_consistent_with_horovod(
            petastorm_reader_kwargs={"cur_shard": 0, "shard_count": 3})

    with mock.patch.dict(os.environ, {'OMPI_COMM_WORLD_RANK': '1', 'OMPI_COMM_WORLD_SIZE': '3'}, clear=True):
        assert (1, 3) == _get_horovod_rank_and_size()
    with mock.patch.dict(os.environ, {'PMI_RANK': '1', 'PMI_SIZE': '3'}, clear=True):
        assert (1, 3) == _get_horovod_rank_and_size()
    with mock.patch.dict(os.environ, {}, clear=True):
        assert (None, None) == _get_horovod_rank_and_size()
        assert _check_rank_and_size_consistent_with_horovod(
            petastorm_reader_kwargs={"cur_shard": 1, "shard_count": 3})


@create_tf_graph
def test_dtype(spark_test_ctx):
    df = spark_test_ctx.spark.range(10)
    df = df.withColumn("float_col", df.id.cast(FloatType())) \
        .withColumn("double_col", df.id.cast(DoubleType()))

    converter1 = make_spark_converter(df)
    with converter1.make_tf_dataset() as dataset:
        iterator = dataset.make_one_shot_iterator()
        tensor = iterator.get_next()
        with tf.Session() as sess:
            ts = sess.run(tensor)
    assert np.float32 == ts.double_col.dtype.type

    converter2 = make_spark_converter(df, dtype='float64')
    with converter2.make_tf_dataset() as dataset:
        iterator = dataset.make_one_shot_iterator()
        tensor = iterator.get_next()
        with tf.Session() as sess:
            ts = sess.run(tensor)
    assert np.float64 == ts.float_col.dtype.type

    converter3 = make_spark_converter(df, dtype=None)
    with converter3.make_tf_dataset() as dataset:
        iterator = dataset.make_one_shot_iterator()
        tensor = iterator.get_next()
        with tf.Session() as sess:
            ts = sess.run(tensor)
    assert np.float32 == ts.float_col.dtype.type
    assert np.float64 == ts.double_col.dtype.type

    with pytest.raises(ValueError, match="dtype float16 is not supported. \
            Use 'float32' or float64"):
        make_spark_converter(df, dtype="float16")


@create_tf_graph
def test_array(spark_test_ctx):
    df = spark_test_ctx.spark.createDataFrame(
        [([1., 2., 3.],),
         ([4., 5., 6.],)],
        StructType([
            StructField(name='c1', dataType=ArrayType(DoubleType()))
        ])
    )
    converter1 = make_spark_converter(df)
    with converter1.make_tf_dataset() as dataset:
        iterator = dataset.make_one_shot_iterator()
        tensor = iterator.get_next()
        with tf.Session() as sess:
            ts = sess.run(tensor)
    assert np.float32 == ts.c1.dtype.type


@pytest.mark.skipif(
    LooseVersion(pyspark.__version__) < LooseVersion("3.0"),
    reason="Vector columns are not supported for pyspark {} < 3.0.0"
    .format(pyspark.__version__))
@create_tf_graph
def test_vector_to_array(spark_test_ctx):
    from pyspark.ml.linalg import Vectors
    from pyspark.mllib.linalg import Vectors as OldVectors
    df = spark_test_ctx.spark.createDataFrame([
        (Vectors.dense(1.0, 2.0, 3.0), OldVectors.dense(10.0, 20.0, 30.0)),
        (Vectors.dense(5.0, 6.0, 7.0), OldVectors.dense(50.0, 60.0, 70.0))
    ], ["vec", "oldVec"])
    converter1 = make_spark_converter(df)
    with converter1.make_tf_dataset(num_epochs=1) as dataset:
        iterator = dataset.make_one_shot_iterator()
        tensor = iterator.get_next()
        with tf.Session() as sess:
            ts = sess.run(tensor)
    assert np.float32 == ts.vec.dtype.type
    assert np.float32 == ts.oldVec.dtype.type
    vec_col = ts.vec[ts.vec[:, 0].argsort()]
    old_vec_col = ts.oldVec[ts.oldVec[:, 0].argsort()]
    assert (2, 3) == ts.vec.shape
    assert (2, 3) == ts.oldVec.shape
    assert ([1., 2., 3.] == vec_col[0]).all() and \
           ([5., 6., 7.] == vec_col[1]).all()
    assert ([10., 20., 30.] == old_vec_col[0]).all() and \
           ([50., 60., 70] == old_vec_col[1]).all()


def test_torch_primitive(spark_test_ctx):
    import torch

    schema = StructType([
        StructField("bool_col", BooleanType(), False),
        StructField("float_col", FloatType(), False),
        StructField("double_col", DoubleType(), False),
        StructField("short_col", ShortType(), False),
        StructField("int_col", IntegerType(), False),
        StructField("long_col", LongType(), False),
        StructField("byte_col", ByteType(), False),
    ])
    df = spark_test_ctx.spark.createDataFrame(
        [(True, 0.12, 432.1, 5, 5, 0, -128),
         (False, 123.45, 0.987, 9, 908, 765, 127)],
        schema=schema).coalesce(1)
    # If we use numPartition > 1, the order of the loaded dataset would
    # be non-deterministic.
    expected_df = df.collect()

    converter = make_spark_converter(df)
    batch = None
    with converter.make_torch_dataloader(num_epochs=1) as dataloader:
        for i, batch in enumerate(dataloader):
            # default batch_size = 1
            for col in df.schema.names:
                actual_ele = batch[col][0]
                expected_ele = expected_df[i][col]
                if col == "float_col" or col == "double_col":
                    # Note that the default dtype is float32
                    assert pytest.approx(expected_ele, rel=1e-6) == actual_ele
                else:
                    assert expected_ele == actual_ele

        assert len(expected_df) == len(converter)
    assert torch.uint8 == batch["bool_col"].dtype
    assert torch.int8 == batch["byte_col"].dtype
    assert torch.float32 == batch["double_col"].dtype
    assert torch.float32 == batch["float_col"].dtype
    assert torch.int32 == batch["int_col"].dtype
    assert torch.int64 == batch["long_col"].dtype
    assert torch.int16 == batch["short_col"].dtype


def test_torch_pickling_remotely(spark_test_ctx):
    df1 = spark_test_ctx.spark.range(100, 101)
    converter1 = make_spark_converter(df1)

    def map_fn(_):
        with converter1.make_torch_dataloader(num_epochs=1) as dataloader:
            for batch in dataloader:
                ret = batch["id"][0]
        return ret

    result = spark_test_ctx.spark.sparkContext.parallelize(range(1), 1) \
        .map(map_fn).collect()[0]
    assert result == 100


def test_torch_batch_size(spark_test_ctx):
    df = spark_test_ctx.spark.range(8)
    conv = make_spark_converter(df)
    batch_size = 2
    with conv.make_torch_dataloader(batch_size=batch_size,
                                    num_epochs=1) as dataloader:
        for batch in dataloader:
            assert batch_size == batch['id'].shape[0]


def test_torch_transform_spec(spark_test_ctx):
    df = spark_test_ctx.spark.range(8)
    conv = make_spark_converter(df)

    from torchvision import transforms
    from petastorm import TransformSpec

    def _transform_row(df_row):
        scale_tranform = transforms.Compose([
            transforms.Lambda(lambda x: x * 0.1),
        ])
        return scale_tranform(df_row)

    transform = TransformSpec(_transform_row)
    with conv.make_torch_dataloader(transform_spec=transform,
                                    num_epochs=1) as dataloader:
        for batch in dataloader:
            assert min(batch['id']) >= 0 and max(batch['id']) < 1


def test_torch_unexpected_param(spark_test_ctx):
    df = spark_test_ctx.spark.range(8)
    conv = make_spark_converter(df)

    with pytest.raises(TypeError, match="unexpected keyword argument 'xyz'"):
        with conv.make_torch_dataloader(xyz=1) as _:
            pass


def test_torch_data_loader_fn(spark_test_ctx):
    from petastorm.pytorch import BatchedDataLoader

    df = spark_test_ctx.spark.range(8)
    conv = make_spark_converter(df)
    with conv.make_torch_dataloader(data_loader_fn=BatchedDataLoader,
                                    batch_size=2,
                                    num_epochs=1) as dataloader:
        assert isinstance(dataloader, BatchedDataLoader)


@mock.patch('petastorm.spark.spark_dataset_converter.make_batch_reader')
def test_torch_dataloader_advanced_params(mock_torch_make_batch_reader, spark_test_ctx):
    SHARD_COUNT = 3
    df = spark_test_ctx.spark.range(100).repartition(SHARD_COUNT)
    conv = make_spark_converter(df)

    mock_torch_make_batch_reader.return_value = \
        make_batch_reader(conv.cache_dir_url)

    with conv.make_torch_dataloader(reader_pool_type='dummy', cur_shard=1,
                                    shard_count=SHARD_COUNT) as _:
        pass
    peta_args = mock_torch_make_batch_reader.call_args[1]
    assert peta_args['reader_pool_type'] == 'dummy' and \
        peta_args['cur_shard'] == 1 and \
        peta_args['shard_count'] == SHARD_COUNT and \
        peta_args['num_epochs'] is None and \
        peta_args['workers_count'] == 4

    # Test default value overridden arguments.
    with conv.make_torch_dataloader(num_epochs=1, workers_count=2) as _:
        pass
    peta_args = mock_torch_make_batch_reader.call_args[1]
    assert peta_args['num_epochs'] == 1 and peta_args['workers_count'] == 2


def test_wait_file_available(spark_test_ctx):
    pq_dir = os.path.join(spark_test_ctx.tempdir, 'test_ev')
    os.makedirs(pq_dir)
    file1_path = os.path.join(pq_dir, 'file1')
    file2_path = os.path.join(pq_dir, 'file2')
    url1 = 'file://' + file1_path.replace(os.sep, '/')
    url2 = 'file://' + file2_path.replace(os.sep, '/')

    url_list = [url1, url2]

    def create_file(p):
        with open(p, 'w'):
            pass

    # 1. test all files exists.
    create_file(file1_path)
    create_file(file2_path)
    _wait_file_available(url_list)

    # 2. test one file does not exists. Raise error.
    os.remove(file2_path)
    with pytest.raises(RuntimeError,
                       match='Timeout while waiting for all parquet-store files to appear at urls'):
        _wait_file_available(url_list)

    # 3. test one file accessible after 1 second.
    def delay_create_file2():
        time.sleep(1)
        create_file(file2_path)

    threading.Thread(target=delay_create_file2()).start()

    _wait_file_available(url_list)


def test_check_dataset_file_median_size(spark_test_ctx, caplog):
    file_size_map = {
        '/a/b/01.parquet': 30,
        '/a/b/02.parquet': 40,
        '/a/b/03.parquet': 50,
        '/a/b/04.parquet': 60,
        '/a/b/05.parquet': 999000,
    }
    with mock.patch('os.path.getsize') as mock_path_get_size:
        mock_path_get_size.side_effect = lambda p: file_size_map[p]
        url_list = ['file://' + path for path in file_size_map.keys()]
        caplog.clear()
        _check_dataset_file_median_size(url_list)
        assert 'The median size' in " ".join(caplog.messages)

        for k in file_size_map:
            file_size_map[k] *= (1024 * 1024)
        caplog.clear()
        _check_dataset_file_median_size(url_list)
        assert 'The median size' not in " ".join(caplog.messages)

        file_size_map = {'/a/b/01.parquet': 29}
        url_list = ['file:///a/b/01.parquet']
        caplog.clear()
        _check_dataset_file_median_size(url_list)
        assert 'The median size' not in " ".join(caplog.messages)


@mock.patch.dict(os.environ, {'DATABRICKS_RUNTIME_VERSION': '7.0'}, clear=True)
def test_check_parent_cache_dir_url(spark_test_ctx, caplog):
    def log_warning_occur():
        return 'you should specify a dbfs fuse path' in '\n'.join([r.message for r in caplog.records])
    with mock.patch('petastorm.spark.spark_dataset_converter._is_spark_local_mode') as mock_is_local:
        mock_is_local.return_value = False
        caplog.clear()
        _check_parent_cache_dir_url('file:/dbfs/a/b')
        assert not log_warning_occur()
        caplog.clear()
        _check_parent_cache_dir_url('file:/a/b')
        assert log_warning_occur()
        mock_is_local.return_value = True
        caplog.clear()
        _check_parent_cache_dir_url('file:/dbfs/a/b')
        assert not log_warning_occur()
        caplog.clear()
        _check_parent_cache_dir_url('file:/a/b')
        assert not log_warning_occur()


def test_get_spark_session_safe_check(spark_test_ctx):
    def map_fn(_):
        _get_spark_session()
        return 0

    with pytest.raises(py4j.protocol.Py4JJavaError):
        spark_test_ctx.spark.sparkContext.parallelize(range(1), 1).map(map_fn).collect()
