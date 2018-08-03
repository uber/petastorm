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

import setuptools
from setuptools import setup
from petastorm import __version__

REQUIRED_PACKAGES = [
    'diskcache>=3.0.0',
    'numpy>=1.13.3',
    'pandas>=0.19.0',
    'pyspark>=2.1.2',
    'pyzmq>=14.0.0',
    # 'pyarrow>=0.10'
]

EXTRA_REQUIRE = {
    'opencv': ['opencv-python>=3.2.0.6'],
    'pyarrow': ['pyarrow>=0.10.0'],
    'tf': ['tensorflow>=1.4.0'],
    'tf_atg': ['atg-tensorflow-gpu>=1.4.0'],
    'tf_gpu': ['tensorflow-gpu>=1.4.0'],
}

packages = setuptools.find_packages()

setup(
    name='petastorm',
    version=__version__,
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    description='petastorm library TODO: more info',
    license='Apache 2.0',
    extras_require=EXTRA_REQUIRE,
)
