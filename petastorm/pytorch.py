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

import collections
import decimal
# Must import pyarrow before torch. See: https://github.com/uber/petastorm/blob/master/docs/troubleshoot.rst
import re

import numpy as np
from six import PY2
from torch.utils.data.dataloader import default_collate

from petastorm.reader_impl.shuffling_buffer import RandomShufflingBuffer, NoopShufflingBuffer

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

    :param dict[str,obj] row_as_dict: a dictionary of key-value pairs. The values types are promoted to
        pytorch compatible.
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
            elif value.dtype == np.bool_:
                row_as_dict[name] = value.astype(np.uint8)
            elif re.search('[SaUO]', value.dtype.str):
                raise TypeError('Pytorch does not support arrays of string classes. Found in field {}'.format(name))
        elif isinstance(value, np.bool_):
            row_as_dict[name] = np.uint8(value)
        elif value is None:
            raise TypeError('Pytorch does not support nullable fields. Found None in {}'.format(name))


def decimal_friendly_collate(batch):
    """A wrapper on top of ``default_collate`` function that allows decimal.Decimal types to be collated.

    We use ``decimal.Decimal`` types in petastorm dataset to represent timestamps. PyTorch's ``default_collate``
    implementation does not support collating ``decimal.Decimal`` types. ``decimal_friendly_collate`` collates
    ``decimal.Decimal`` separately and then combines with the rest of the fields collated by a standard
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

    This loader can be used as an iterator and will terminate when the reader used in the construction of the class
    runs out of samples.
    """

    def __init__(self, reader, batch_size=1, collate_fn=decimal_friendly_collate,
                 shuffling_queue_capacity=0):
        """
        Initializes a data loader object, with a default collate.

        Number of epochs is defined by the configuration of the reader argument.

        An optional shuffling queue is created if shuffling_queue_capacity is greater than 0. No samples will be
        returned to a user by the ``DataLoader`` until the queue is full. After that, batches of `batch_size`
        will be created by uniformly sampling the shuffling queue. Once no more samples are available from the data
        reader, the shuffling queue is allowed to be consumed till no further samples are available.

        Note that the last returned batch could have less then ``batch_size`` samples.

        NOTE: if you are using ``make_batch_reader``, this shuffling queue will be randomizing the order of the
        entire batches and not changing the order of elements within a batch. This is likely not what you intend to do.

        :param reader: petastorm Reader instance
        :param batch_size: the number of items to return per batch; factored into the len() of this reader
        :param collate_fn: an optional callable to merge a list of samples to form a mini-batch.
        :param shuffling_queue_capacity: Queue capacity is passed to the underlying :class:`tf.RandomShuffleQueue`
          instance. If set to 0, no suffling will be done.
        """
        self.reader = reader
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        if shuffling_queue_capacity > 0:
            # We always add and remove one sample from the shuffling buffer in the steady state. min_after_dequeue
            # is important when enqueue and dequeue rates are not the same.
            # By setting `min_after_dequeue = shuffling_queue_capacity - 1` we will just keep the buffer full.
            min_after_dequeue = shuffling_queue_capacity - 1
            self._shuffling_buffer = RandomShufflingBuffer(shuffling_queue_capacity,
                                                           min_after_retrieve=min_after_dequeue, extra_capacity=0)
        else:
            self._shuffling_buffer = NoopShufflingBuffer()

    def __iter__(self):
        """
        The Data Loader iterator stops the for-loop when reader runs out of samples.
        """
        batch = []
        for row in self.reader:
            # Default collate does not work nicely on namedtuples and treat them as lists
            # Using dict will result in the yielded structures being dicts as well
            row_as_dict = row._asdict()

            # Promote some types that are incompatible with pytorch to be pytorch friendly.
            _sanitize_pytorch_types(row_as_dict)

            # Add rows to shuffling buffer
            self._shuffling_buffer.add_many([row_as_dict])

            # If the shuffling buffer is ready, pull the resampled rows out and try add to our accumulated batch
            while self._shuffling_buffer.can_retrieve():
                post_shuffled_row = self._shuffling_buffer.retrieve()
                batch.append(post_shuffled_row)

                # Batch is ready? Collate and emmit
                if len(batch) == self.batch_size:
                    yield self.collate_fn(batch)
                    batch = []

        # Once reader can not emit new rows, we might still have a bunch of rows waiting in the shuffling buffer.
        # Telling shuffling buffer thatt we are finished allows to deplete the buffer completely, regardless its
        # min_after_dequeue setting.
        self._shuffling_buffer.finish()

        # Some code duplication with the main reading loop. Not sure how to reorganize to avoid this without
        # overcomplicating the code too much.
        while self._shuffling_buffer.can_retrieve():
            post_shuffled_row = self._shuffling_buffer.retrieve()
            batch.append(post_shuffled_row)

            # Batch is ready? Collate and emmit
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
