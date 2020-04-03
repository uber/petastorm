import tempfile

from distutils.version import LooseVersion

import pyspark
import pytest

from examples.spark_dataset_converter.utils import download_mnist_libsvm
from petastorm.spark import spark_dataset_converter


@pytest.fixture(scope='module')
def mnist_dir():
    tmp_dir = tempfile.mkdtemp('_converter_example_test')
    download_mnist_libsvm(tmp_dir)
    return tmp_dir


@pytest.mark.skipif(
    LooseVersion(pyspark.__version__) < LooseVersion("3.0"),
    reason="Vector columns are not supported for pyspark {} < 3.0.0"
    .format(pyspark.__version__))
def test_converter_pytorch_example(mnist_dir):
    from examples.spark_dataset_converter.pytorch_converter_example import run
    # Clean internal _parent_cache_dir_url setting so that this test can set different one
    spark_dataset_converter._parent_cache_dir_url = None
    run(mnist_dir)


@pytest.mark.skipif(
    LooseVersion(pyspark.__version__) < LooseVersion("3.0"),
    reason="Vector columns are not supported for pyspark {} < 3.0.0"
    .format(pyspark.__version__))
def test_converter_tf_example(mnist_dir):
    from examples.spark_dataset_converter.tensorflow_converter_example import run
    # Clean internal _parent_cache_dir_url setting so that this test can set different one
    spark_dataset_converter._parent_cache_dir_url = None
    run(mnist_dir)
