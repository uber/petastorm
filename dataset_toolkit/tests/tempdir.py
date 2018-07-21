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

import contextlib
import os
import shutil
import tempfile


@contextlib.contextmanager
def temporary_directory(*args, **kwargs):
    """Create and return the path to a temporary directory

    Return a path to a newly created temporary directory with visibility
    in the file system. The root directory must exist; parents of
    the temporary directory will neither be created nor destroyed. The
    created directory (and any of its contents) will be automatically deleted
    when the context is exited.

        with tempfile.temporary_directory() as f:
            ...

    :param dir: The directory to root the temporary directory under
    :yields: The path to the temporary directory
    """
    path = None
    try:
        path = tempfile.mkdtemp(*args, **kwargs)
        yield path
    finally:
        if path and os.path.exists(path):
            shutil.rmtree(path)
