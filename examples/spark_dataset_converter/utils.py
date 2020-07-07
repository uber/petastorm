import os
import tempfile

import requests


def download_mnist_libsvm(mnist_data_dir):
    mnist_data_path = os.path.join(mnist_data_dir, "mnist.bz2")
    data_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2"
    r = requests.get(data_url)
    with open(mnist_data_path, "wb") as f:
        f.write(r.content)


def get_mnist_dir():
    # This folder is baked into the docker image
    MNIST_DATA_DIR = "/data/mnist/"

    if os.path.isdir(MNIST_DATA_DIR) and os.path.isfile(os.path.join(MNIST_DATA_DIR, 'mnist.bz2')):
        return MNIST_DATA_DIR

    mnist_dir = tempfile.mkdtemp('_mnist_data')
    download_mnist_libsvm(mnist_dir)
    return mnist_dir
