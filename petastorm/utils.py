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
from decimal import Decimal
from multiprocessing import Pool

import numpy as np
import pyarrow
from future.utils import raise_with_traceback
from pyarrow.filesystem import LocalFileSystem

logger = logging.getLogger(__name__)


def run_in_subprocess(func, *args, **kwargs):
    """
    Run some code in a separate process and return the result. Once the code is done, terminate the process.
    This prevents a memory leak in the other process from affecting the current process.

    Gotcha: func must be a functioned defined at the top level of the module.
    :param kwargs: dict
    :param args: list
    :param func:
    :return:
    """
    pool = Pool(1)
    result = pool.apply(func, args=args, kwds=kwargs)

    # Probably not strictly necessary since terminate is called on GC, but it's not guaranteed when the pool will get
    # GC'd.
    pool.terminate()
    return result


class DecodeFieldError(RuntimeError):
    pass


def decode_row(row, schema):
    """
    Decode dataset row according to coding spec from unischema object

    If a codec is set, we use codec.decode() to produce decoded value.

    For scalar fields, the codec maybe set to `None`. In that case:
     - If the numpy_dtype is a numpy scalar or a Decimal, cast the 'encoded' value before returning.
     - In any other case, return the value 'as is'.

    :param row: dictionary with encodded values
    :param schema: unischema object
    :return:
    """
    decoded_row = dict()
    for field_name_unicode, _ in row.items():
        field_name = str(field_name_unicode)
        if field_name in schema.fields:
            try:
                if row[field_name] is not None:
                    field = schema.fields[field_name]
                    codec = schema.fields[field_name].codec
                    if codec:
                        decoded_row[field_name] = codec.decode(field, row[field_name])
                    elif field.numpy_dtype and issubclass(field.numpy_dtype, (np.generic, Decimal)):
                        decoded_row[field_name] = field.numpy_dtype(row[field_name])
                    else:
                        decoded_row[field_name] = row[field_name]
                else:
                    decoded_row[field_name] = None
            except Exception:  # pylint: disable=broad-except
                raise_with_traceback(DecodeFieldError('Decoding field "{}" failed'.format(field_name)))

    return decoded_row


def add_to_dataset_metadata(dataset, key, value):
    """
    Adds a key and value to the parquet metadata file of a parquet dataset.
    :param dataset: (ParquetDataset) parquet dataset
    :param key:     (str) key of metadata entry
    :param value:   (str) value of metadata
    """
    if not isinstance(dataset.paths, str):
        raise ValueError('Expected dataset.paths to be a single path, not a list of paths')

    metadata_file_path = dataset.paths.rstrip('/') + '/_metadata'
    common_metadata_file_path = dataset.paths.rstrip('/') + '/_common_metadata'
    common_metadata_file_crc_path = dataset.paths.rstrip('/') + '/._common_metadata.crc'

    # If the metadata file already exists, add to it.
    # Otherwise fetch the schema from one of the existing parquet files in the dataset
    if dataset.fs.exists(common_metadata_file_path):
        with dataset.fs.open(common_metadata_file_path) as f:
            arrow_metadata = pyarrow.parquet.read_metadata(f)
    elif dataset.fs.exists(metadata_file_path):
        # If just the metadata file exists and not the common metadata file, copy the contents of
        # the metadata file to the common_metadata file for backwards compatibility
        with dataset.fs.open(metadata_file_path) as f:
            arrow_metadata = pyarrow.parquet.read_metadata(f)
    else:
        arrow_metadata = dataset.pieces[0].get_metadata()

    base_schema = arrow_metadata.schema.to_arrow_schema()

    # base_schema.metadata may be None, e.g.
    metadata_dict = base_schema.metadata or dict()
    metadata_dict[key] = value
    schema = base_schema.with_metadata(metadata_dict)

    with dataset.fs.open(common_metadata_file_path, 'wb') as metadata_file:
        pyarrow.parquet.write_metadata(schema, metadata_file)

    # We have just modified _common_metadata file, but the filesystem implementation used by pyarrow does not
    # update the .crc value. We better delete the .crc to make sure there is no mismatch between _common_metadata
    # content and the checksum.
    if isinstance(dataset.fs, LocalFileSystem) and dataset.fs.exists(common_metadata_file_crc_path):
        try:
            dataset.fs.rm(common_metadata_file_crc_path)
        except NotImplementedError:
            os.remove(common_metadata_file_crc_path)
