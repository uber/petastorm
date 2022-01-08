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
import sys

import dill

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # An entry point to the newely executed process.
    # Will unpickle function handle and arguments and call the function.
    try:
        logging.basicConfig()
        if len(sys.argv) != 2:
            raise RuntimeError('Expected a single command line argument')
        new_process_runnable_file = sys.argv[1]

        with open(new_process_runnable_file, 'rb') as f:
            func, args, kargs = dill.load(f)  # pylint: disable=unpacking-non-sequence

        # Don't need the pickle file with the runable. Cleanup.
        os.remove(new_process_runnable_file)

        func(*args, **kargs)
    except Exception as e:
        logger.error('Unhandled exception in the function launched by exec_in_new_process: %s', str(e))
        raise
