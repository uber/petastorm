#
# Uber, Inc. (c) 2018
#
"""This setup script should be launched by running
source/python/av/ml/dataset_toolkit/setup/package.sh"""

import setuptools
from setuptools import setup
from dataset_toolkit import __version__

REQUIRED_PACKAGES = [
    'numpy>=1.13.3',
    'Pillow>=3.0',
    'pyspark>=2.1.2',
    'pyzmq>=14.0.0',
    'tensorflow>=1.4',
    'pyarrow>=0.8',
    'pandas>=0.19.0',
    'diskcache>=3.0.0',
]

packages = setuptools.find_packages()

setup(
    name='dataset_toolkit',
    version=__version__,
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    description='Dataset toolkit',
    license='TBD',
)
