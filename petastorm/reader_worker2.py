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

import hashlib

from pyarrow import parquet as pq
from pyarrow.parquet import ParquetFile
from six.moves.urllib.parse import urlparse, urlunparse

from petastorm import utils
from petastorm.cache import NullCache
from petastorm.fs_utils import FilesystemResolver
from petastorm.workers_pool.worker_base import WorkerBase


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


def _filter_keys(dictionary, keys_to_include):
    return {key: dictionary[key] for key in keys_to_include}


class Loader(object):
    def __init__(self, worker_id, dataset_url, schema, sequence, split_pieces, local_cache, worker_predicate):
        self._dataset_url_parsed = urlparse(dataset_url)
        self._schema = schema
        self._sequence = sequence
        self._split_pieces = split_pieces
        self._local_cache = local_cache
        self._worker_predicate = worker_predicate

        # We create datasets lazily in the first invocation of 'def process'. This speeds up startup time since
        # all Worker constructors are serialized
        resolver = FilesystemResolver(self._dataset_url_parsed)
        self._dataset = pq.ParquetDataset(
            resolver.parsed_dataset_url().path,
            filesystem=resolver.filesystem(),
            validate_schema=False)

    def load(self, row_group):

        # Create pyarrow file system
        parquet_file = ParquetFile(self._dataset.fs.open(row_group.path))

        if self._worker_predicate:
            if not isinstance(self._local_cache, NullCache):
                raise RuntimeError('Local cache is not supported together with predicates, '
                                   'unless the dataset is partitioned by the column the predicate operates on.')
            all_cols = self._load_rows_with_predicate(parquet_file, row_group, self._worker_predicate)
        else:
            # Using hash of the dataset url with the relative path in order to:
            #  1. Make sure if a common cache serves multiple processes (e.g. redis), we don't have conflicts
            #  2. Dataset url is hashed, to make sure we don't create too long keys, which maybe incompatible with
            #     some cache implementations
            #  3. Still leave relative path and the piece_index in plain text to make it easier to debug
            cache_key = '{}:{}:{}'.format(hashlib.md5(urlunparse(self._dataset_url_parsed).encode('utf-8')).hexdigest(),
                                          row_group.path, row_group)
            all_cols = self._local_cache.get(cache_key, lambda: self._load_rows(parquet_file, row_group))

        all_cols_as_tuples = all_cols

        if self._sequence:
            all_cols_as_tuples = self._sequence.form_sequence(data=all_cols_as_tuples)

        return all_cols_as_tuples

    def _load_rows(self, file, piece):
        """Loads all rows from a piece"""

        # pyarrow would fail if we request a column names that the dataset is partitioned by, so we strip them from
        # the `columns` argument.
        partitions = self._dataset.partitions
        column_names = set(field.name for field in self._schema.fields.values()) - partitions.partition_names

        all_rows = piece.read(open_file_func=lambda _: file, columns=column_names, partitions=partitions) \
            .to_pandas() \
            .to_dict('records')

        return all_rows

    def _load_rows_with_predicate(self, file, piece, worker_predicate):
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
        predicate_rows = piece.read(
            open_file_func=lambda _: file,
            columns=predicate_column_names,
            partitions=self._dataset.partitions).to_pandas().to_dict('records')

        # Decode values
        decoded_predicate_rows = predicate_rows

        # Use the predicate to filter
        match_predicate_mask = [worker_predicate.do_include(_filter_keys(row, predicate_column_names)) for row in decoded_predicate_rows]

        # Don't have anything left after filtering? Exit early.
        if not any(match_predicate_mask):
            return []

        # Remove rows that were filtered out by the predicate
        filtered_decoded_predicate_rows = [row for i, row in enumerate(decoded_predicate_rows) if
                                           match_predicate_mask[i]]

        if other_column_names:
            # Read remaining columns
            other_rows = piece.read(
                open_file_func=lambda filepath: file,
                columns=other_column_names,
                partitions=self._dataset.partitions).to_pandas().to_dict('records')

            # Remove rows that were filtered out by the predicate
            filtered_other_rows = [row for i, row in enumerate(other_rows) if match_predicate_mask[i]]

            # Decode remaining columns
            decoded_other_rows = filtered_other_rows

            # Merge predicate needed columns with the remaining
            all_cols = [_merge_two_dicts(a, b) for a, b in zip(decoded_other_rows, filtered_decoded_predicate_rows)]
            return all_cols
        else:
            return filtered_decoded_predicate_rows

