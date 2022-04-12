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


import io
import re

import setuptools
from setuptools import setup

PACKAGE_NAME = 'petastorm'

with open('README.rst') as f:
    long_description = f.read()

with io.open('petastorm/__init__.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)
    if version is None:
        raise ImportError('Could not find __version__ in petastorm/__init__.py')

REQUIRED_PACKAGES = [
    'dill>=0.2.1',
    'diskcache>=3.0.0',
    'future>=0.10.2',
    'numpy>=1.13.3',
    'packaging>=15.0',
    'pandas>=0.19.0',
    'psutil>=4.0.0',
    'pyspark>=2.1.0',
    'pyzmq>=14.0.0',
    'pyarrow>=0.17.1',
    'six>=1.5.0',
    'fsspec',
]

EXTRA_REQUIRE = {
    # `docs` versions are to facilitate local generation of documentation.
    # Sphinx 1.3 would be desirable, but is incompatible with current ATG setup.
    # Thus the pinning of both sphinx and alabaster versions.
    'docs': [
        'sphinx>=1.2.2',
        'alabaster>=0.7.11'
    ],
    'opencv': ['opencv-python>=3.2.0.6'],
    's3fs': ['s3fs>=0.5.0'],
    'tf': ['tensorflow>=1.14.0'],
    'tf_gpu': ['tensorflow-gpu>=1.14.0'],
    'test': [
        'Pillow>=6.2.1',
        'codecov>=2.0.15',
        'flake8',
        'gcsfs>=0.2.0',
        'mock>=2.0.0',
        'mypy',
        'opencv-python>=3.2.0.6',
        'pylint>=1.9',
        'pytest>=3.0.0',
        'pytest-cov>=2.5.1',
        'pytest-forked>=0.2',
        'pytest-logger>=0.4.0',
        'pytest-timeout>=1.3.3',
        'requests>=2.22.0',
        's3fs>=0.0.1',
        'gcsfs',
        "types-pytz==2021.1.0",
        "types-six==0.1.7",
    ],
    'torch': ['torchvision>=0.2.1', 'torch>=1.5.0'],
}

packages = setuptools.find_packages()

setup(
    name=PACKAGE_NAME,
    version=version,
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    description='Petastorm is a library enabling the use of Parquet storage from Tensorflow, Pytorch, and'
                ' other Python-based ML training frameworks.',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    license='Apache License, Version 2.0',
    extras_require=EXTRA_REQUIRE,
    entry_points={
        'console_scripts': [
            'petastorm-copy-dataset.py=petastorm.tools.copy_dataset:main',
            'petastorm-generate-metadata.py=petastorm.etl.petastorm_generate_metadata:main',
            'petastorm-throughput.py=petastorm.benchmark.cli:main',
        ],
    },
    url='https://github.com/uber/petastorm',
    author='Uber Technologies, Inc.',
    python_requires='>=3',
    classifiers=[
        'Environment :: Console',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
