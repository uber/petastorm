import os

import requests


def download_mnist_libsvm(mnist_data_dir):
    mnist_data_path = os.path.join(mnist_data_dir, "mnist.bz2")
    data_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2"
    r = requests.get(data_url)
    with open(mnist_data_path, "wb") as f:
        f.write(r.content)
