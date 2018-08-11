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
    'six>=1.5.0'
    # 'pyarrow>=0.10'
]

if '--minimal-deps' not in sys.argv:
    REQUIRED_PACKAGES += [
        # 'opencv-python>=3.2.0.6',
        # 'pyarrow>=0.10',
    ]
else:
    PACKAGE_NAME += '_min_deps'
    sys.argv.remove('--minimal-deps')


EXTRA_REQUIRE = {
    'tf': ['tensorflow>=1.4.0'],
    'tf_gpu': ['tensorflow-gpu>=1.4.0'],
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
)
