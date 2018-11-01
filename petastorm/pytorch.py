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

# Must import pyarrow before torch. See: https://github.com/uber/petastorm/blob/master/docs/troubleshoot.rst
import pyarrow  # noqa: F401 pylint: disable=W0611

import collections
import decimal

import numpy as np
from six import PY2
from torch.utils.data.dataloader import default_collate

if PY2:
    _string_classes = basestring  # noqa: F821
else:
    _string_classes = (str, bytes)


def _sanitize_pytorch_types(row_as_dict):
    """Promotes values types in a dictionary to the types supported by pytorch. Raises an error if type is clear error
    if the type can not be promoted.

    The parameter is modified in-place.

    int8, uint16 are promoted to int32; uint32 -> int64;
    numpy string_, unicode_, object arrays are not supported.

    :param row_as_dict: a dictinoary of key-value pairs. The values types are promoted to pytorch compatible.
    :return: None
    """
    for name, value in row_as_dict.items():
        # PyTorch supported types are: double, float, float16, int64, int32, and uint8
        if isinstance(value, np.ndarray):
            if value.dtype == np.int8:
                row_as_dict[name] = value.astype(np.int16)
            elif value.dtype == np.uint16:
                row_as_dict[name] = value.astype(np.int32)
            elif value.dtype == np.uint32:
                row_as_dict[name] = value.astype(np.int64)
        if isinstance(value, np.bool_):
            row_as_dict[name] = np.uint8(value)


def decimal_friendly_collate(batch):
    """A wrapper on top of ``default_collate`` function that allows decimal.Decimal types to be collated.

    We use ``decimal.Decimal`` types in petastorm dataset to represent timestamps. PyTorch's ``default_collate``
    implementation does not support collating ``decimal.Decimal`` types. ``decimal_friendly_collate`` collates
    ``decimal.Decima al`` separately and then combines with the rest of the fields collated by a standard
    ``default_collate``.

    :param batch: A list of dictionaries to collate
    :return: A dictionary of lists/pytorch.Tensor types
    """

    if isinstance(batch[0], decimal.Decimal):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: decimal_friendly_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], _string_classes):
        return batch
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [decimal_friendly_collate(samples) for samples in transposed]
    else:
        return default_collate(batch)


class DataLoader(object):
    """
    A data loader adaptor for ``torch.utils.data.DataLoader``.

    This class iterates and returns items from the Reader in batches.

    This loader can be used as a context manager, but it will terminate at the end of an epoch.
    The context will invoke next_epoch() upon entry.

    If not used as context manager, invoke the next_epoch() function at the start of each epoch, and
    once more at the very end.
    """

    def __init__(self, reader, batch_size=1, collate_fn=decimal_friendly_collate, transform=None):
        """
        Initializes a data loader object, with a default collate and optional transform functions.

        This loader handles multiple epochs by instantiating a new Reader per epoch.

        :param reader: petastorm Reader instance
        :param batch_size: the number of items to return per batch; factored into the len() of this reader
        :param collate_fn: a optional callable to merge a list of samples to form a mini-batch.
        :param transform: a optional tranform function to apply to each data row
        """
        self.reader = reader
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.transform = transform

    def __iter__(self):
        """
        The Data Loader iterator stops the for-loop at the end of each epoch, but a subsequent for-loop
        will instantiate a new Reader and yield more results, until the requested number of epoch has been
        reached.  After that point, any subsequent call results in StopIteration, per iterator protocol.
        """
        batch = []
        for row in self.reader:
            # Default collate does not work nicely on namedtuples and treat them as lists
            # Using dict will result in the yielded structures being dicts as well
            row_as_dict = row._asdict()
            _sanitize_pytorch_types(row_as_dict)
            transformed_row = self.transform(row_as_dict) if self.transform else row_as_dict
            batch.append(transformed_row)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if batch:
            yield self.collate_fn(batch)

    # Functions needed to treat data loader as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.stop()
        self.reader.join()
