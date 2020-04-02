import os
import subprocess


def download_mnist_libsvm():
    data_location = "/tmp/petastorm/mnist/mnist.bz2"
    if os.path.exists(data_location):
        return
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sh_path = os.path.join(dir_path, "..", "download_mnist_libsvm.sh")
    retcode = subprocess.call([sh_path])
    assert retcode == 0


def test_converter_pytorch_example():
    from examples.spark_dataset_converter.pytorch_single_node import main
    download_mnist_libsvm()
    main()


def test_converter_tf_single_example():
    from examples.spark_dataset_converter.tensorflow_single_node import main
    download_mnist_libsvm()
    main()


def test_converter_tf_worker_example():
    from examples.spark_dataset_converter.tensorflow_spark_worker import main
    download_mnist_libsvm()
    main()
