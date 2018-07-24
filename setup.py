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
    'numpy>=1.13.3',
    'Pillow>=3.0',
    'pyspark>=2.1.2',
    'pyzmq>=14.0.0',
    # 'pyarrow>=0.8',  # Temporarary removing dependency on pyarrow - we'll pick up whatever we have in ATG repo
                       # Once there is 0.10, we'll be able to use stock, non-atg version.
    'pandas>=0.19.0',
    'diskcache>=3.0.0',
]

EXTRA_REQUIRE = {
    'tf': ['tensorflow>=1.4.0'],
    'tf_gpu': ['tensorflow-gpu>=1.4.0'],
    'tf_atg': ['atg-tensorflow-gpu>=1.4.0'],
    'pyarrow': ['pyarrow>=0.8'],  # TODO(yevgeni): once ATG goes to stock 0.10 (instead of atg-pyarrow=0.9.0.1), we
                                  # make the require mandatory.
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
