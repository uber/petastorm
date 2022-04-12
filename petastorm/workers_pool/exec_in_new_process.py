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
import logging
import os
import subprocess
import sys
from tempfile import mkstemp

import dill

logger = logging.getLogger(__name__)


def exec_in_new_process(func, *args, **kargs):
    """Launches a function in a separate process. Takes variable number of arguments which are passed to the function.
    The process IS NOT FORKED by 'exec'ed.

    :param func: Function to be executed in a separate process.
    :param args: position arguments passed to the func
    :param kargs: named arguments passed to the func
    :return:
    """

    # Store function handle and arguments into a pickle
    new_process_runnable_handle, new_process_runnable_file = mkstemp(suffix='runnable')
    with os.fdopen(new_process_runnable_handle, 'wb') as f:
        dill.dump((func, args, kargs), f)

    bootstrap_package_name = '{}.{}'.format(__package__, os.path.splitext(os.path.basename(__file__))[0])
    # Popen this script (__main__) below will be an entry point
    process = subprocess.Popen(args=[sys.executable,
                                     '-m',
                                     bootstrap_package_name + "_entrypoint",
                                     new_process_runnable_file],
                               executable=sys.executable)
    return process
