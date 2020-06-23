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
            return not(len(contents) == 1 and contents[0] == path)
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

        for obj in self.fs.ls(path, detail=True):
            # each info name must be at least [path]/part , but here
            # we check also for names like [path]/part/
            obj_path = obj['name']
            if obj_path.strip('/') == path.strip('/'):
                continue
            if obj['type'] == 'directory':
                directories.add(obj_path)
            elif obj['type'] == 'file':
                files.add(obj_path)

        rel_files = sorted([posixpath.split(f)[1] for f in files
                            if f not in directories])
        rel_directories = sorted([posixpath.split(x[:-1])[1]
                                  for x in directories])

        yield path, rel_directories, rel_files

        for directory in directories:
            for tup in self.walk(directory):
                yield tup
