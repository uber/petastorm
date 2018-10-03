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
from io import BytesIO

import numpy as np


class RowStorageBundler(object):
    def __init__(self, schema):
        """RowStorageBundler handles bundling a row's fields into a smaller number columns for storage.
        This is used to prevent wasting a round trip to remote storage to read a very small amount of data
        which is specifically a problem with Parquet if your rows are a mix of very large and very small
        columns. Your row group size is determined by the large columns, meaning the small column will contain
        very little data in a single row group.

        :param schema: A Unischema instance
        """
        self._schema = schema

    def bundle_row(self, row):
        """
        Takes a dictionary of encoded fields and bundles them into columns for storage. It encodes the
        bundles as a numpy array.

        Example:

        Input:
          row = {
            'rgb-image': bytearray(),
            'image-id': 'my-image-id',
            'height-px': 240,
            'width-px': 240,
            ...
          }

        Output:
          row = {
            'image-bundle': bytearray(),
            ...
          }


        :param row:
        :return: a dictionary of the bundled fields
        """
        bundled_dict = {}
        included_field_names = set()
        for bundle_name, bundle_columns in self._schema.column_bundles.items():
            bundle_vals = [row[field_name] for field_name in bundle_columns]
            numpy_array = np.array(bundle_vals, dtype=object)
            memfile = BytesIO()
            np.save(memfile, numpy_array)
            bundled_dict[bundle_name] = bytearray(memfile.getvalue())
            memfile.close()
            # Keep track of included fields to add any columns not included in a bundle individually
            included_field_names.update(bundle_columns)
        for field_name in row:
            if field_name not in included_field_names:
                bundled_dict[field_name] = row[field_name]
        return bundled_dict

    def debundle_row(self, row):
        """
        Converts a row with bundled columns to a row with all fields flattened properly.
        All fields in the returned dictionary are still individually encoded.

        Example:

        Input:
          row = {
            'image-bundle': bytearray(),
            ...
          }

        Output:
          row = {
            'rgb-image': bytearray(),
            'image-id': 'my-image-id',
            'height-px': 240,
            'width-px': 240,
            ...
          }

        :param row: A dictionary of columns, some of which are potentially bundled
        :return: A dictionary of encoded unischema fields
        """
        unbundled_row = dict()
        for column_name_unicode, column_value in row.items():
            column_name = str(column_name_unicode)
            if column_name in self._schema.column_bundles:
                memfile = BytesIO(column_value)
                decoded_bundle = np.load(memfile)
                memfile.close()
                bundled_fields = self._schema.column_bundles[column_name]
                for i, field in enumerate(bundled_fields):
                    unbundled_row[field] = decoded_bundle[i]
            else:
                unbundled_row[column_name] = column_value
        return unbundled_row
