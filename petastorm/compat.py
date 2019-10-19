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

"""0.15.0 cancelled previously deprecated functions. We still want to support 0.11 as it is being used by some users.
This file implements compatibility interfaces. Once we drop support of 0.11, we can get rid of this file."""

import pyarrow as pa
from packaging import version
from pyarrow import parquet as pq

_PYARROW_BEFORE_013 = version.parse(pa.__version__) < version.parse('0.13.0')


def compat_get_metadata(piece, open_func):
    if _PYARROW_BEFORE_013:
        arrow_metadata = piece.get_metadata(open_func)
    else:
        arrow_metadata = piece.get_metadata()
    return arrow_metadata


def compat_piece_read(piece, open_file_func, **kwargs):
    if _PYARROW_BEFORE_013:
        table = piece.read(open_file_func=open_file_func, **kwargs)
    else:
        table = piece.read(**kwargs)
    return table


def compat_table_columns_gen(table):
    if _PYARROW_BEFORE_013:
        for column in table.columns:
            name = column.name
            yield name, column
    else:
        for name in table.column_names:
            column = table.column(name)
            yield name, column


def compat_column_num_chunks(column):
    if _PYARROW_BEFORE_013:
        return column.data.num_chunks
    else:
        return column.num_chunks


def compat_make_parquet_piece(path, open_file_func, **kwargs):
    if _PYARROW_BEFORE_013:
        return pq.ParquetDatasetPiece(path, **kwargs)
    else:
        return pq.ParquetDatasetPiece(path, open_file_func=open_file_func,   # pylint: disable=unexpected-keyword-arg
                                      **kwargs)
