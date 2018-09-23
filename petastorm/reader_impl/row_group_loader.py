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

from __future__ import division

import hashlib

import numpy as np
from pyarrow import parquet as pq
from pyarrow.parquet import ParquetFile
from six.moves.urllib.parse import urlparse, urlunparse

from petastorm.cache import NullCache
from petastorm.fs_utils import FilesystemResolver


def _merge_two_dicts(a, b):
    """Merges two dictionaries together. If the same key is present in both input dictionaries, the value from 'b'
    dominates."""
    result = a.copy()
    result.update(b)
    return result


def _select_cols(a_dict, keys):
    """Filteres out entries in a dictionary that have a key which is not part of 'keys' argument. `a_dict` is not
    modified and a new dictionary is returned."""
    if keys == list(a_dict.keys()):
        return a_dict
    else:
        return {field_name: a_dict[field_name] for field_name in keys}


class RowGroupLoader(object):
    def __init__(self, dataset_url, schema, ngram, local_cache, worker_predicate):
        """RowGroupLoader responsible for loading one rowgroup at a time. Rows returned are returned encoded.

        :param dataset_url: A url of a parquet dataset.
        :param schema: A unischema corresponding to the data in the dataset
        :param ngram: An instance of NGram if ngrams should be read or None, if each row in the dataset corresponds to
          a single sample returned.
        :param local_cache: An instance of a rowgroup cache (CacheBase interface) object to be used.
        :param worker_predicate: An instance of predicate (PredicateBase interface)
        """
        self._dataset_url_parsed = urlparse(dataset_url)
        self._schema = schema
        self._ngram = ngram
        self._local_cache = local_cache
        self._worker_predicate = worker_predicate

        resolver = FilesystemResolver(self._dataset_url_parsed)
        self._dataset = pq.ParquetDataset(
            resolver.parsed_dataset_url().path,
            filesystem=resolver.filesystem(),
            validate_schema=False)

    def load(self, rowgroup_spec):
        """Loads data form a single rowgroup from the dataset.

        Reads a single rowgroup from a dataset. Returns a list of dictionary with still encoded data.
        If worker_predicate was passed to the constructor, the predicate is first applied to the columns specified
        by the predicate. The rest of the columns are loaded only if at least one row matches the predicate.

        A rowgroup will be loaded from local cache, if cache contains an instance of the rowgroup.

        If ngram not None was passed to the constructor, the function returns a dictionary structured according to
        NGram definition.

        :param rowgroup_spec: A dictionary containing the following fields: 'row_group': ParquetDatasetPiece object
          describing a rowgroup to be loaded; 'shuffle_row_drop_partition' a tuple with
          (this_partition, num_of_partitions)
        :return: A dictionary indexed by field names, or a dictionary defined by NGram spec.
        """
        piece = rowgroup_spec['row_group']
        shuffle_row_drop_partition = rowgroup_spec['shuffle_row_drop_partition']

        # Create pyarrow file system
        with self._dataset.fs.open(piece.path) as piece_file_handle:
            parquet_file = ParquetFile(piece_file_handle)

            if not isinstance(self._local_cache, NullCache):
                if self._worker_predicate:
                    raise RuntimeError('Local cache is not supported together with predicates, '
                                       'unless the dataset is partitioned by the column the predicate operates on.')
                if shuffle_row_drop_partition[1] != 1:
                    raise RuntimeError('Local cache is not supported together with shuffle_row_drop_partitions > 1')

            if self._worker_predicate:
                all_cols = self._load_rows_with_predicate(parquet_file, piece, self._worker_predicate,
                                                          shuffle_row_drop_partition)
            else:
                # Using hash of the dataset url with the relative path in order to:
                #  1. Make sure if a common cache serves multiple processes (e.g. redis), we don't have conflicts
                #  2. Dataset url is hashed, to make sure we don't create too long keys, which maybe incompatible with
                #     some cache implementations
                #  3. Still leave relative path and the piece_index in plain text to make it easier to debug
                cache_key = '{}:{}:{}'.format(
                    hashlib.md5(urlunparse(self._dataset_url_parsed).encode('utf-8')).hexdigest(),
                    piece.path, piece.row_group)
                all_cols = self._local_cache.get(cache_key,
                                                 lambda: self._load_rows(parquet_file, piece,
                                                                         shuffle_row_drop_partition))

        if self._ngram:
            all_cols_as_ngrams = self._ngram.form_ngram(data=all_cols, schema=self._schema)
            return all_cols_as_ngrams
        else:
            return all_cols

    def _load_rows(self, parquet_file, piece, shuffle_row_drop_range):
        """Loads all rows from a piece"""

        # pyarrow would fail if we request a column names that the dataset is partitioned by, so we strip them from
        # the `columns` argument.
        partitions = self._dataset.partitions
        column_names = set(field.name for field in self._schema.fields.values()) - partitions.partition_names

        all_rows = self._read_with_shuffle_row_drop(piece, parquet_file, column_names, shuffle_row_drop_range)

        return all_rows

    def _load_rows_with_predicate(self, parquet_file, piece, worker_predicate, shuffle_row_drop_partition):
        """Loads all rows that match a predicate from a piece"""

        # 1. Read all columns needed by predicate and decode
        # 2. Apply the predicate. If nothing matches, exit early
        # 3. Read the remaining columns and decode
        # 4. Combine with columns already decoded for the predicate.

        # Split all column names into ones that are needed by predicateand the rest.
        predicate_column_names = set(worker_predicate.get_fields())

        if not predicate_column_names:
            raise ValueError('At least one field name must be returned by predicate\'s get_field() method')

        all_schema_names = set(field.name for field in self._schema.fields.values())

        invalid_column_names = predicate_column_names - all_schema_names
        if invalid_column_names:
            raise ValueError('At least some column names requested by the predicate ({}) '
                             'are not valid schema names: ({})'.format(', '.join(invalid_column_names),
                                                                       ', '.join(all_schema_names)))

        other_column_names = all_schema_names - predicate_column_names - \
            self._dataset.partitions.partition_names

        # Read columns needed for the predicate
        predicate_rows = self._read_with_shuffle_row_drop(piece, parquet_file, predicate_column_names,
                                                          shuffle_row_drop_partition)

        # Decode values
        decoded_predicate_rows = [_select_cols(row, predicate_column_names) for row in predicate_rows]

        # Use the predicate to filter
        match_predicate_mask = [worker_predicate.do_include(row) for row in decoded_predicate_rows]

        # Don't have anything left after filtering? Exit early.
        if not any(match_predicate_mask):
            return []

        # Remove rows that were filtered out by the predicate
        filtered_decoded_predicate_rows = [row for i, row in enumerate(decoded_predicate_rows) if
                                           match_predicate_mask[i]]

        if other_column_names:
            # Read remaining columns
            other_rows = self._read_with_shuffle_row_drop(piece, parquet_file, other_column_names,
                                                          shuffle_row_drop_partition)

            # Remove rows that were filtered out by the predicate
            filtered_other_rows = [row for i, row in enumerate(other_rows) if match_predicate_mask[i]]

            # Decode remaining columns
            decoded_other_rows = filtered_other_rows

            # Merge predicate needed columns with the remaining
            all_cols = [_merge_two_dicts(a, b) for a, b in zip(decoded_other_rows, filtered_decoded_predicate_rows)]
            return all_cols
        else:
            return filtered_decoded_predicate_rows

    def _read_with_shuffle_row_drop(self, piece, parquet_file, column_names, shuffle_row_drop_partition):
        data_frame = piece.read(
            open_file_func=lambda _: parquet_file,
            columns=column_names,
            partitions=self._dataset.partitions
        ).to_pandas()

        num_rows = len(data_frame)
        num_partitions = shuffle_row_drop_partition[1]
        this_partition = shuffle_row_drop_partition[0]

        partition_indexes = np.floor(np.arange(num_rows) / (float(num_rows) / min(num_rows, num_partitions)))

        if self._ngram:
            # If we have an ngram we need to take elements from the next partition to build the ngram
            next_partition_indexes = np.where(partition_indexes >= this_partition + 1)
            if next_partition_indexes[0].size:
                next_partition_to_add = next_partition_indexes[0][0:self._ngram.length - 1]
                partition_indexes[next_partition_to_add] = this_partition

        selected_dataframe = data_frame.loc[partition_indexes == this_partition]
        return selected_dataframe.to_dict('records')
