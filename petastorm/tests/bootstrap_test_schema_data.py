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
from __future__ import print_function

import getopt
import sys

from petastorm.tests.test_common import create_test_dataset


# Number of rows in a fake dataset
ROWS_COUNT = 10


def usage_exit(msg=None):
    if msg:
        print(msg)
    print("""\
Usage: {} [options]

Options:
  -h, --help           Show this message
  --output-dir <dir>   Path of directory where to write test data
""".format(sys.argv[0]))
    sys.exit(1)


def make_test_metadata(path):
    """
    Use test_common to make a dataset for the TestSchema.

    :param path: path to store the test dataset
    :return: resulting dataset as a dictionary
    """
    assert path, 'Please supply a nonempty path to store test dataset.'
    return create_test_dataset('file://{}'.format(path), range(ROWS_COUNT))


if __name__ == '__main__':
    try:
        options, args = getopt.getopt(sys.argv[1:], 'ho:', ['--help', 'output-dir='])
        path = None
        for opt, value in options:
            if opt in ('-h', '--help'):
                usage_exit()
            if opt in ('-o', '--output-dir'):
                if value:
                    path = value
        if path is None or not path == 0:
            usage_exit('Please supply an output directory.')
        else:
            make_test_metadata(path)
    except getopt.GetoptError as msg:
        usage_exit(msg)
