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
import logging
# Must import pyarrow before torch. See: https://github.com/uber/petastorm/blob/master/docs/troubleshoot.rst
import re

import numpy as np
import torch
from contextlib import contextmanager
from packaging import version
from six import PY2
from torch.utils.data.dataloader import default_collate

from petastorm.reader import make_batch_reader, make_reader
from petastorm.reader_impl.pytorch_shuffling_buffer import BatchedRandomShufflingBuffer, \
    BatchedRandomShufflingBufferWithMemCache, BatchedNoopShufflingBuffer
from petastorm.reader_impl.shuffling_buffer import RandomShufflingBuffer, NoopShufflingBuffer

_TORCH_BEFORE_1_1 = version.parse(torch.__version__) < version.parse('1.1.0')  # type: ignore

if PY2:
    _string_classes = basestring  # noqa: F821
else:
    _string_classes = (str, bytes)

logger = logging.getLogger(__name__)


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
            if value.dtype == np.int8 and _TORCH_BEFORE_1_1:
                row_as_dict[name] = value.astype(np.int16)
            elif value.dtype == np.uint16:
                row_as_dict[name] = value.astype(np.int32)
            elif value.dtype == np.uint32:
                row_as_dict[name] = value.astype(np.int64)
            elif value.dtype == np.bool_:
                row_as_dict[name] = value.astype(np.uint8)
            elif re.search('[SaUO]', value.dtype.str):
                raise TypeError('Pytorch does not support arrays of string or object classes. '
                                'Found in field {}.'.format(name))
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


_PARALLEL_ITER_ERROR = "You must finish a full pass of Petastorm DataLoader before making another pass from the \
beginning.If you do need to terminate early and restart from beginning, please re-create the reader and the data \
loader."


@contextmanager
def make_torch_reader_and_loader(dataset_url_or_urls,
                                 num_epochs=1,
                                 batch_size=1,
                                 transform_fn=None,
                                 shuffling_queue_capacity=0,
                                 in_memory_cache_size=0,
                                 transform_spec=None,
                                 **kwargs):
    num_epochs_to_load = None
    if in_memory_cache_size:
        # When caching in memory, reader reads the data from source only once. Rest of the epochs
        # are served from in-memory cached buffer in the loader.
        num_epochs_to_load = num_epochs
        num_epochs = 1

    # In general, make_batch_reader is faster than make_reader for reading the dataset.
    # However, we found out that make_reader performs data transformations much faster than
    # make_batch_reader with parallel worker processes. Therefore, the default reader
    # we choose is make_batch_reader unless there are data transformations.
    if transform_spec:
        reader_factory = make_reader
        if 'pyarrow_serialize' not in kwargs:
            kwargs['pyarrow_serialize'] = True
    else:
        reader_factory = make_batch_reader

    reader = reader_factory(dataset_url_or_urls, num_epochs=num_epochs, **kwargs)
    try:
        loader = BatchedDataLoader(reader,
                                   batch_size=batch_size,
                                   transform_fn=transform_fn,
                                   shuffling_queue_capacity=shuffling_queue_capacity,
                                   in_memory_cache_size=in_memory_cache_size,
                                   num_epochs_to_load=num_epochs_to_load)

        yield loader
    finally:
        reader.stop()
        reader.join()


class LoaderBase(object):

    def __init__(self):
        self._in_iter = None
        self._error = None

    def __iter__(self):
        if self._error is not None:
            raise RuntimeError('Cannot start a new iteration because last time iteration failed with error {err}.'
                               .format(err=repr(self._error)))
        if self._in_iter is not None and self._in_iter == True:  # noqa: E712
            raise RuntimeError(_PARALLEL_ITER_ERROR)
        if self._in_iter is not None:
            self.reader.reset()
            logger.warning('Start a new pass of Petastorm DataLoader, reset underlying Petastorm reader to position 0.')
        self._in_iter = True

        try:
            for batch in self._iter_impl():
                yield batch
        except Exception as e:
            self._error = e
            logger.error('Iteration on Petastorm DataLoader raise error: %s', repr(e))
            raise
        finally:
            self._in_iter = False


class DataLoader(LoaderBase):
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
        super(DataLoader, self).__init__()
        self.reader = reader
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        # _batch_acc accumulates samples for a single batch.
        self._batch_acc = []
        self.shuffling_queue_capacity = shuffling_queue_capacity
        self._in_iter = None

    def _iter_impl(self):
        """
        The Data Loader iterator stops the for-loop when reader runs out of samples.
        """
        # As we iterate over incoming samples, we are going to store them in `self._batch_acc`, until we have a batch of
        # the requested batch_size ready.

        keys = None
        if self.shuffling_queue_capacity > 0:
            # We can not know what is the reasonable number to use for the extra capacity, so we set a huge number
            # and give up on the unbound growth protection mechanism.
            min_after_dequeue = self.shuffling_queue_capacity - 1
            self._shuffling_buffer = RandomShufflingBuffer(self.shuffling_queue_capacity,
                                                           min_after_retrieve=min_after_dequeue,
                                                           extra_capacity=100000000)
        else:
            self._shuffling_buffer = NoopShufflingBuffer()

        for row in self.reader:
            # Default collate does not work nicely on namedtuples and treat them as lists
            # Using dict will result in the yielded structures being dicts as well
            row_as_dict = row._asdict()

            keys = row_as_dict.keys()

            # Promote some types that are incompatible with pytorch to be pytorch friendly.
            _sanitize_pytorch_types(row_as_dict)

            # Add rows to shuffling buffer
            if not self.reader.batched_output:
                self._shuffling_buffer.add_many([row_as_dict])
            else:
                # Transposition:
                #   row_as_dict:        {'a': [1,2,3], 'b':[4,5,6]}
                #   row_group_as_tuple: [(1, 4), (2, 5), (3, 6)]
                # The order within a tuple is defined by key order in 'keys'
                row_group_as_tuple = list(zip(*(row_as_dict[k] for k in keys)))

                # Adding data as 'row-by-row' into a shuffling buffer. This is a pretty
                # slow implementation though. Probably can comeup with a faster way to shuffle,
                # perhaps at the expense of a larger memory consumption...
                self._shuffling_buffer.add_many(row_group_as_tuple)

            # _yield_batches will emit as much batches as are allowed by the shuffling_buffer (RandomShufflingBuffer
            # will avoid underflowing below a certain number of samples to guarantee some samples decorrelation)
            for batch in self._yield_batches(keys):
                yield batch

        # Once reader can not read new rows, we might still have a bunch of rows waiting in the shuffling buffer.
        # Telling shuffling buffer that we are finished allows to deplete the buffer completely, regardless its
        # min_after_dequeue setting.
        self._shuffling_buffer.finish()

        for batch in self._yield_batches(keys):
            yield batch

        # Yield the last and partial batch
        if self._batch_acc:
            yield self.collate_fn(self._batch_acc)

    def _yield_batches(self, keys):
        while self._shuffling_buffer.can_retrieve():
            post_shuffled_row = self._shuffling_buffer.retrieve()
            if not isinstance(post_shuffled_row, dict):
                # This is for the case of batched reads. Here we restore back the
                # dictionary format of records
                post_shuffled_row = dict(zip(keys, post_shuffled_row))

            self._batch_acc.append(post_shuffled_row)

            # Batch is ready? Collate and emmit
            if len(self._batch_acc) == self.batch_size:
                yield self.collate_fn(self._batch_acc)
                self._batch_acc = []

    # Functions needed to treat data loader as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.stop()
        self.reader.join()


class BatchedDataLoader(LoaderBase):
    """
    Same as DataLoader except it uses torch-based shuffling buffers which enable batched buffering
    (significantly faster for small data).
    """

    def __init__(self, reader, batch_size=1,
                 transform_fn=None,
                 shuffling_queue_capacity=0,
                 in_memory_cache_size=0,
                 num_epochs_to_load=None):
        """
        Initializes a data loader object.

        Number of epochs is defined by the configuration of the reader argument.

        An optional shuffling queue is created if shuffling_queue_capacity is greater than 0. No samples will be
        returned to a user by the ``BatchedDataLoader`` until the queue is full. After that, batches of `batch_size`
        will be created by uniformly sampling the shuffling queue. Once no more samples are available from the data
        reader, the shuffling queue is allowed to be consumed till no further samples are available.

        Note that the last returned batch could have less then ``batch_size`` samples.

        NOTE: if you are using ``make_batch_reader``, this shuffling queue will be randomizing the order of the
        entire batches and not changing the order of elements within a batch. This is likely not what you intend to do.

        This class does not support special types that are not supported in PyTorch (decimal/string).

        :param reader: petastorm Reader instance
        :param batch_size: the number of items to return per batch; factored into the len() of this reader
        :param transform_fn: an optional callable to convert batches from the reader to PyTorch tensors
        :param shuffling_queue_capacity: Queue capacity is passed to the underlying :class:`tf.RandomShuffleQueue`
          instance. If set to 0, no shuffling will be done.
        :param in_memory_cache_size: This is an integer that indicates the size of in-memory cache.
            Note that the data will be cached in the Shuffling buffer and not in the
            reader itself because we do not want to cache it twice. Currently, only
            BatchedDataLoader supports in-memory caching. If set to 0, in memory-caching will be
            disabled.
        :param num_epochs_to_load: Number of epochs to load when in memory cache is enabled. If
            set to None, loader will loads batches indefinitely.
        """
        super(BatchedDataLoader, self).__init__()
        self.reader = reader
        self.batch_size = batch_size
        self.transform_fn = transform_fn or torch.as_tensor

        # _batch_acc accumulates samples for a single batch.
        self._batch_acc = []
        self.shuffling_queue_capacity = shuffling_queue_capacity
        self._in_iter = None

        self.in_memory_cache_size = in_memory_cache_size

        if self.in_memory_cache_size > 0 and self.reader.num_epochs_to_read != 1:
            raise ValueError("When cache in loader memory is activated, reader.num_epochs_to_read "
                             "must be set to 1. When caching the data in memory, we need to read "
                             "the data only once. The rest of the time, we serve it from memory.")

        # This is only relevant if in memory caching is activated.
        self.num_epochs_to_load = num_epochs_to_load
        if self.in_memory_cache_size == 0 and self.num_epochs_to_load:
            raise ValueError("num_epochs_to_load needs to be specified when "
                             "cache_in_loader_memory is enabled.")

        if self.in_memory_cache_size < 0:
            raise ValueError("Cannot create a in-memory cache. cache_size_limit larger than zero.")

        if self.shuffling_queue_capacity > 0 and self.in_memory_cache_size:
            raise ValueError("When using in-memory cache, shuffling_queue_capacity has no effect.")

    def _iter_impl(self):
        """
        The Data Loader iterator stops the for-loop when reader runs out of samples.
        """
        # As we iterate over incoming samples, we are going to store them in `self._batch_acc`, until we have a batch of
        # the requested batch_size ready.

        keys = None

        if self.in_memory_cache_size:
            self._shuffling_buffer = BatchedRandomShufflingBufferWithMemCache(
                cache_size=self.in_memory_cache_size,
                num_epochs_to_load=self.num_epochs_to_load,
                batch_size=self.batch_size
            )
        elif self.shuffling_queue_capacity > 0:
            # We can not know what is the reasonable number to use for the extra capacity, so we set a huge number
            # and give up on the unbound growth protection mechanism.
            # To keep the same behavior as DataLoader, we need to increase the shuffling_queue_capacity
            min_after_dequeue = self.shuffling_queue_capacity - 1
            shuffling_queue_capacity = min_after_dequeue + self.batch_size
            self._shuffling_buffer = BatchedRandomShufflingBuffer(
                shuffling_queue_capacity,
                min_after_retrieve=min_after_dequeue,
                extra_capacity=100000000,
                batch_size=self.batch_size,
            )
        else:
            self._shuffling_buffer = BatchedNoopShufflingBuffer(batch_size=self.batch_size)

        for row in self.reader:
            # Default collate does not work nicely on namedtuples and treat them as lists
            # Using dict will result in the yielded structures being dicts as well
            row_as_dict = row._asdict()

            keys = row_as_dict.keys()

            # Promote some types that are incompatible with pytorch to be pytorch friendly.
            _sanitize_pytorch_types(row_as_dict)

            # Add rows to shuffling buffer
            for k, v in row_as_dict.items():
                if not self.reader.batched_output:
                    row_as_dict[k] = self.transform_fn([v])
                else:
                    row_as_dict[k] = self.transform_fn(v)
            self._shuffling_buffer.add_many(row_as_dict.values())

            # _yield_batches will emit as much batches as are allowed by the shuffling_buffer (RandomShufflingBuffer
            # will avoid underflowing below a certain number of samples to guarantee some samples decorrelation)
            for batch in self._yield_batches(keys):
                yield batch

        # When caching memory, at this point we are done with reading all the data. Rest of the
        # batches will be created directly from memory. Otherwise if there is not in memory cache,
        # Once reader can not read new rows, we might still have a bunch of rows waiting in the
        # shuffling buffer. Telling shuffling buffer that we are finished allows to deplete the
        # buffer completely, regardless its min_after_dequeue setting.
        self._shuffling_buffer.finish()

        print("Fardin Abdi done with caching")

        for batch in self._yield_batches(keys):
            yield batch

    def _yield_batches(self, keys):
        while self._shuffling_buffer.can_retrieve():
            batch = self._shuffling_buffer.retrieve()
            if not isinstance(batch, dict):
                # This is for the case of batched reads. Here we restore back the
                # dictionary format of records
                batch = dict(zip(keys, batch))
            yield batch

    # Functions needed to treat data loader as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.stop()
        self.reader.join()
