#  Copyright (c) 2017-2020 Uber Technologies, Inc.
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

import posixpath

from pyarrow.filesystem import FileSystem, DaskFileSystem
from pyarrow.util import implements, _stringify_path


class GCSFSWrapper(DaskFileSystem):

    @implements(FileSystem.isdir)
    def isdir(self, path):
        from gcsfs.core import norm_path
        path = norm_path(_stringify_path(path))
        try:
            contents = self.fs.ls(path)
            if len(contents) == 1 and contents[0] == path:
                return False
            else:
                return True
        except OSError:
            return False

    @implements(FileSystem.isfile)
    def isfile(self, path):
        from gcsfs.core import norm_path
        path = norm_path(_stringify_path(path))
        try:
            contents = self.fs.ls(path)
            return len(contents) == 1 and contents[0] == path
        except OSError:
            return False

    def walk(self, path):
        """
        Directory tree generator, like os.walk

        Generator version of what is in gcsfs, which yields a flattened list of
        files
        """
        from gcsfs.core import norm_path
        path = norm_path(_stringify_path(path))
        directories = set()
        files = set()

        for key in self.fs.ls(path, detail=True):
            # each info name must be at least [path]/part , but here
            # we check also for names like [path]/part/
            path = key['name']
            if key['storageClass'] == 'DIRECTORY':
                if path.endswith('/'):
                    directories.add(path[:-1])
                else:
                    directories.add(path)
            elif key['storageClass'] == 'BUCKET':
                pass
            else:
                files.add(path)

        files = sorted([posixpath.split(f)[1] for f in files
                        if f not in directories])
        directories = sorted([posixpath.split(x)[1]
                              for x in directories])

        yield path, directories, files

        for directory in directories:
            for tup in self.walk(directory):
                yield tup
