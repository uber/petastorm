#
# Uber, Inc. (c) 2018
#
"""A copy of `temporary_directory`. Prevents creating dependency on ATG's av package"""

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
