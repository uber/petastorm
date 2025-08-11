#  Copyright (c) 2022 BlackBerry Limited.
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

from pyarrow import parquet as pq


class PetastormPyArrowDataset:

    def __init__(self, path_or_paths, filesystem=None, validate_schema=None, filters=None, metadata_nthreads=None,
                 use_new_arrow_api=False):
        if use_new_arrow_api:
            raise NotImplementedError("The implementation using the new PyArrow API is not yet complete")

        kwargs = {}
        if filesystem:
            kwargs["filesystem"] = filesystem
        if validate_schema is not None:
            kwargs["validate_schema"] = validate_schema
        if filters:
            kwargs["filters"] = filters
        if metadata_nthreads:
            kwargs["metadata_nthreads"] = metadata_nthreads

        self._legacy_dataset = pq.ParquetDataset(path_or_paths, **kwargs)

    @property
    def common_metadata(self):
        return self._legacy_dataset.common_metadata

    @property
    def metadata(self):
        return self._legacy_dataset.metadata

    @property
    def metadata_path(self):
        return self._legacy_dataset.metadata_path

    @property
    def fs(self):
        return self._legacy_dataset.fs

    @property
    def partitions(self):
        return self._legacy_dataset.partitions

    @property
    def paths(self):
        return self._legacy_dataset.paths

    @property
    def pieces(self):
        if not hasattr(self, "_wrapped_pieces"):
            self._wrapped_pieces = [_wrap_piece(piece) for piece in self._legacy_dataset.pieces]
        return self._wrapped_pieces


def _wrap_piece(piece):
    return PetastormPyArrowDatasetPiece(piece.path, open_file_func=piece.open_file_func, row_group=piece.row_group,
                                        partition_keys=piece.partition_keys)


class PetastormPyArrowDatasetPiece:

    def __init__(self, path, open_file_func, row_group, partition_keys, use_new_arrow_api=False):
        if use_new_arrow_api:
            raise NotImplementedError("The implementation using the new PyArrow API is not yet complete")

        self._legacy_piece = pq.ParquetDatasetPiece(path, open_file_func=open_file_func, row_group=row_group,
                                                    partition_keys=partition_keys)

    def get_metadata(self):
        return self._legacy_piece.get_metadata()

    def read(self, *, columns, partitions):
        return self._legacy_piece.read(columns=columns, partitions=partitions)

    @property
    def path(self):
        return self._legacy_piece.path

    @property
    def partition_keys(self):
        return self._legacy_piece.partition_keys

    @property
    def row_group(self):
        return self._legacy_piece.row_group
