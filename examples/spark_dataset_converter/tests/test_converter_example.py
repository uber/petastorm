from distutils.version import LooseVersion

import pyspark
import pytest

# This folder is baked into the docker image
MNIST_DATA_DIR = "/data/mnist/"


@pytest.mark.skipif(
    LooseVersion(pyspark.__version__) < LooseVersion("3.0"),
    reason="Vector columns are not supported for pyspark {} < 3.0.0"
    .format(pyspark.__version__))
def test_converter_pytorch_example():
    from examples.spark_dataset_converter.pytorch_converter_example import run
    run(MNIST_DATA_DIR)


@pytest.mark.skipif(
    LooseVersion(pyspark.__version__) < LooseVersion("3.0"),
    reason="Vector columns are not supported for pyspark {} < 3.0.0"
    .format(pyspark.__version__))
def test_converter_tf_example():
    from examples.spark_dataset_converter.tensorflow_converter_example import run
    run(MNIST_DATA_DIR)
