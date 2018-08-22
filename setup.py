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
import sys

import setuptools
from setuptools import setup
from petastorm import __version__

PACKAGE_NAME = 'petastorm'

REQUIRED_PACKAGES = [
    'diskcache>=3.0.0',
    'numpy>=1.13.3',
    'pandas>=0.19.0',
    'pyspark>=2.1.0',
    'pyzmq>=14.0.0',
    'pyarrow>=0.10',
    'six>=1.5.0',
]

EXTRA_REQUIRE = {
    'opencv': ['opencv-python>=3.2.0.6'],
    'tf': ['tensorflow>=1.4.0'],
    'tf_gpu': ['tensorflow-gpu>=1.4.0'],
    'test' : ['opencv-python>=3.2.0.6', 'pytest>=3.0.0', 'pytest-cov>=2.5.1',
              'Pillow>=3.0', 'pylint>=1.9', 'codecov>=2.0.15'],
}

packages = setuptools.find_packages()

setup(
    name=PACKAGE_NAME,
    version=__version__,
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    description='petastorm library TODO: more info',
    license='Apache 2.0',
    extras_require=EXTRA_REQUIRE,
    entry_points = {
        'console_scripts': ['petastorm-generate-metadata=petastorm.etl.petastorm_generate_metadata:main'],
    },
)
