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

"""A set of Tensorflow specific helper functions for the unischema"""
import datetime
import sys
import warnings
from calendar import timegm
from collections import OrderedDict, namedtuple
from decimal import Decimal

import numpy as np
import tensorflow as tf

# Mapping of identical datatypes in numpy-ish and tensorflow-ish
_NUMPY_TO_TF_DTYPES_MAPPING = {
    np.bool: tf.bool,
    np.int8: tf.int8,
    np.int16: tf.int16,
    np.int32: tf.int32,
    np.int64: tf.int64,
    np.uint8: tf.uint8,
    np.uint16: tf.int32,
    np.uint32: tf.int64,
    np.float32: tf.float32,
    np.float64: tf.float64,
    np.string_: tf.string,
    np.unicode_: tf.string,
    np.str_: tf.string,
    np.bool_: tf.bool,
    Decimal: tf.string,
    np.datetime64: tf.int64,
}

# Name of an op in the TF graph used for the random shuffling queue. This name can be used by diagnostics code that
# wishes to read-out shuffling queue size
RANDOM_SHUFFLING_QUEUE_SIZE = 'random_shuffling_queue_size'


def date_to_nsec_from_epoch(dt):
    return timegm(dt.timetuple()) * 1000000000


_date_to_nsec_from_epoch_vectorized = np.vectorize(date_to_nsec_from_epoch)


def _sanitize_field_tf_types(sample):
    """Takes a named tuple and casts/promotes types unknown to TF to the types that are known.

    Three casts that are currently implemented
      - Decimal to string
      - uint16 to int32
      - np.datetime64 to int64, as nanoseconds since unix epoch

    :param sample: named tuple or a dictionary
    :return: same type as the input with values casted to types supported by Tensorflow
    """
    next_sample_dict = sample._asdict()

    for k, v in next_sample_dict.items():
        if v is None:
            raise RuntimeError('Encountered "{}"=None. Tensorflow does not support None values as a tensor.'
                               'Consider filtering out these rows using a predicate.'.format(k))
        # Assuming conversion to the same numpy type is trivial and dirty cheap
        if isinstance(v, Decimal):
            # Normalizing decimals only to get rid of the trailing zeros (makes testing easier, assuming has
            # no other effect)
            next_sample_dict[k] = str(v.normalize())
        elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.datetime64):
            # Convert to nanoseconds from POSIX epoch
            next_sample_dict[k] = (v - np.datetime64('1970-01-01T00:00:00.0')) \
                .astype('timedelta64[ns]').astype(np.int64)
        elif isinstance(v, np.ndarray) and v.dtype == np.uint16:
            next_sample_dict[k] = v.astype(np.int32)
        elif isinstance(v, np.ndarray) and v.dtype == np.uint32:
            next_sample_dict[k] = v.astype(np.int64)
        elif isinstance(v, np.ndarray) and v.dtype.type in (np.bytes_, np.unicode_):
            if v.size != 0:
                next_sample_dict[k] = v.tolist()
        elif isinstance(v, np.ndarray) and v.dtype.kind == 'O' and isinstance(v[0], datetime.date):
            # Pyarrow 0.12.1 started returning python datetime.date when parquet column is a DateType() column.
            # Convert values in such column into nsec from epoch int64.
            next_sample_dict[k] = _date_to_nsec_from_epoch_vectorized(v)

    # Construct object of the same type as the input
    return sample.__class__(**next_sample_dict)


def _schema_to_tf_dtypes(schema):
    """Returns schema as a list of tensorflow dtypes.
    :param schema: The schema.
    :return: List of tensorflow dtypes.
    """
    return [_numpy_to_tf_dtypes(f.numpy_dtype) for f in schema.fields.values()]


def _schema_to_tf_dtypes_ngram(schema, ngram):
    """Returns schema as a list of tensorflow dtypes for a ngram.
    :param schema: The schema.
    :param ngram: The ngram.
    :return: tensorflow dtypes for a ngram.
    """
    result = []
    # Iterate over each timestep
    for key in sorted(ngram.fields.keys()):
        # Get schema at that timestep
        new_schema = ngram.get_schema_at_timestep(schema=schema, timestep=key)
        for field in new_schema.fields.values():
            result.append(_numpy_to_tf_dtypes(field.numpy_dtype))
    return result


def _numpy_to_tf_dtypes(numpy_dtype):
    """Returns a tensorflow dtype object corresponding to numpy's dtype.

    A :class:`ValueError` is raised if there is no known mapping between the types

    :param numpy_dtype: numpy dtype object
    :return: tensorflow dtype object
    """
    if numpy_dtype in _NUMPY_TO_TF_DTYPES_MAPPING:
        if numpy_dtype == np.unicode_ and sys.version_info >= (3, 0):
            warnings.warn("Tensorflow will convert all unicode strings back to bytes type. "
                          "You may need to decode values.", UnicodeWarning)
        return _NUMPY_TO_TF_DTYPES_MAPPING[numpy_dtype]
    else:
        raise ValueError('Unknown mapping of numpy {} to tensorflow dtype'.format(numpy_dtype))


def _flatten(data):
    """Flattens the data, where it takes a dictionary of timesteps, each value is a dictionary and converts it to
    one flat dictionary having a key that is the key of the inner dictionary + '_' + timestep.

    For example, ``data`` would be ``{1: {'a': 'avalue', 'b': 'bvalue'}, 2: {'c': 'cvalue', 'd': 'dvalue'}}`` and the
    output of :func:`._flatten` would be ``{'a_1': 'avalue', 'b_1': 'bvalue', 'c_2': 'cvalue', 'd_2': 'dvalue'}``.

    :param data: The data to flatten.
    :return: The flattened dictionary.
    """
    flattened = OrderedDict()
    for index, key in enumerate(sorted(data.keys())):
        data_dict = data[key]._asdict()
        for subkey in data_dict:
            encoded_key = subkey + '_' + str(index)
            flattened[encoded_key] = data_dict[subkey]

    FlattenedTuple = namedtuple('flattened', list(flattened.keys()))
    return FlattenedTuple(**flattened)


def make_namedtuple_tf_ngram(unischema, ngram, *args, **kargs):
    """Creates a dictionary of timestep keys and namedtuple values from args and kargs.

    :param ngram: The ngram definition.
    :param args: args.
    :param kargs: kargs.
    :return: A dictionary of timestep keys and namedtuple values.
    """

    ngram_result = {}
    previous_args_end = 0
    for timestep in range(min(ngram.fields.keys()), max(ngram.fields.keys()) + 1):
        # For each timestep iteration, mark the args and kargs for that timestep and create
        # a namedtuple from them.
        current_field_names = ngram.get_field_names_at_timestep(timestep)
        new_schema = ngram.get_schema_at_timestep(schema=unischema, timestep=timestep)
        new_args_end = previous_args_end + len(current_field_names)
        args_timestep = args[previous_args_end:new_args_end]
        previous_args_end = new_args_end
        kargs_timestep = (kargs[str(timestep)] if str(timestep) in kargs else {})
        ngram_result[timestep] = new_schema._get_namedtuple()(*args_timestep, **kargs_timestep)
    return ngram_result


def _set_shape(schema, fields_as_dict, batched_output=None):
    # Assign static shape for all tensors
    # Workaround of an issue described here:
    # https://stackoverflow.com/questions/49161316/trailing-x00-characters-in-tensor-when-numpy-string-array-is-returned-from-tf
    for k in fields_as_dict.keys():
        unischema_field = schema.fields[k]

        if batched_output:
            shape = (None,) + unischema_field.shape
        else:
            shape = unischema_field.shape
        # Set static shape
        fields_as_dict[k].set_shape(shape)


def _shuffling_queue(shuffling_queue_capacity, min_after_dequeue, dtypes, fields_as_list):
    """Creates a shuffling queue with enqueue/dequeue pair. Always a single writing thread."""

    # Named tuples loose the 'named' part when going via queue
    shuffling_queue = tf.RandomShuffleQueue(shuffling_queue_capacity, min_after_dequeue, dtypes)

    # The following call to .size has a side effect of creating a new node in the TF graph. We are interested
    # in the side effect so we can read the queue size somewhere else, addressing the node by a 'well-known-name'
    shuffling_queue.size(name=RANDOM_SHUFFLING_QUEUE_SIZE)

    # We need the queue only for shuffling, so we use only a single enqueuing thread (actually would be happy
    # not to introduce any threads. Not sure if there is such a mechanism in TF)
    queue_runner = tf.train.QueueRunner(shuffling_queue, 1 * [shuffling_queue.enqueue(fields_as_list)])

    tf.train.add_queue_runner(queue_runner)

    # Passed through the queue. We got an ordered list. The order matches the order of fields in unischema
    fields_as_list = shuffling_queue.dequeue()
    return fields_as_list


def _tf_tensors_nonngram(reader, shuffling_queue_capacity, min_after_dequeue):
    """A tensorflow data adapter for non ngrams. Return value is a named tuple with tensorflow tensors supplying
    the data directly into a Tensoflow graph. See `tf_tensor` documentation for input/output arguments meaning."""

    # TODO(yevgeni): implement a mechanism for signaling that we have no more data
    def dequeue_sample_impl(x):
        next_sample = next(reader)
        # Decimal is not supported by TF. int8,16,32,64 scalars are all returned as python native int type
        # (casted to 64 bit by tensorflow). sanitize_field_tf_types will explicitly convert all values
        # to explicit numpy types making it compatible with return values expected by Tensorflow
        return _sanitize_field_tf_types(next_sample)

    # fields_as_list is a list with tensors matching the order of the values in the schema. named-tuple semantics is
    # not preserved across tf.py_func call boundary.
    fields_as_list = tf.py_func(dequeue_sample_impl, [tf.constant(1)], _schema_to_tf_dtypes(reader.schema))

    if shuffling_queue_capacity > 0:
        # Pass py_func output via shuffling queue if requested.
        fields_as_list = _shuffling_queue(shuffling_queue_capacity, min_after_dequeue,
                                          _schema_to_tf_dtypes(reader.schema), fields_as_list)

    # Going via `make_namedtuple_tf` is a little wasteful, since we are converting directly to dict. However, this
    # spares the need to implement a function similar to make_namedtuple_tf that returns dict instead of a named tuple
    fields_as_dict = reader.schema.make_namedtuple_tf(*fields_as_list)._asdict()

    # Force all static shapes to be set in the returned value based on the unischema
    _set_shape(reader.schema, fields_as_dict, reader.batched_output)

    # Make a row tensor into a nice named tuple
    return reader.schema.make_namedtuple_tf(**fields_as_dict)


def _tf_tensors_ngram(reader, shuffling_queue_capacity, min_after_dequeue):
    """A tensorflow data adapter for ngrams. Return value is a named tuple with tensorflow tensors supplying
    the data directly into a Tensoflow graph. See `tf_tensor` documentation for input/output arguments meaning."""

    # TODO(yevgeni): implement a mechanism for signaling that we have no more data
    def dequeue_sample_impl(x):
        next_sample = next(reader)
        assert isinstance(next_sample, dict)

        # Create a dictionary, where each key is a timestep, and value is named tuple or dictionary.
        ngram = {}
        for timestep in next_sample:
            ngram[timestep] = _sanitize_field_tf_types(next_sample[timestep])

        return _flatten(ngram)

    fields_as_list = tf.py_func(dequeue_sample_impl, [tf.constant(1)],
                                _schema_to_tf_dtypes_ngram(reader.schema, reader.ngram))

    if shuffling_queue_capacity > 0:
        # Pass py_func output via shuffling queue if requested.
        fields_as_list = _shuffling_queue(shuffling_queue_capacity, min_after_dequeue,
                                          _schema_to_tf_dtypes_ngram(reader.schema, reader.ngram), fields_as_list)

    fields_as_namedtuple = make_namedtuple_tf_ngram(reader.schema, reader.ngram, *fields_as_list)

    # We change the key to str format here in order to be able to use ** later to expand the dictionary as kargs.
    fields_as_dict = {
        str(timestep): fields_as_namedtuple[timestep]._asdict() for timestep in fields_as_namedtuple}
    for timestep in fields_as_dict:
        _set_shape(reader.schema, fields_as_dict[timestep])

    return make_namedtuple_tf_ngram(reader.schema, reader.ngram, **fields_as_dict)


def tf_tensors(reader, shuffling_queue_capacity=0, min_after_dequeue=0):
    """Bridges between python-only interface of the Reader (next(Reader)) and tensorflow world.

    This function returns a named tuple of tensors from the dataset, e.g.,

    >>> row_tensors
    >>> Out[2]: TestSchema_view(field_1=<tf.Tensor 'PyFunc:0' shape=() dtype=string>,
    >>>         field_2=<tf.Tensor 'StringSplit:1' shape=(?,) dtype=string>,
    >>>         field_3=<tf.Tensor 'PyFunc:2' shape=() dtype=int64>, ...)

    If the reader was created with ``ngram=NGram(...)`` parameter, then a dictionary of named tuples is returned
    (indexed by time):

    >>> row_tensors
    >>> Out[6]:
    >>> {0: TestSchema_view(field_1=<tf.Tensor 'PyFunc_4:0' shape=() dtype=string>, field_2=...),
    >>>  1: TestSchema_view(field_1=<tf.Tensor 'PyFunc_4:11' shape=() dtype=string>, field_2=...),
    >>>  2: TestSchema_view(field_1=<tf.Tensor 'PyFunc_4:22' shape=() dtype=string>, field_2=...)}

    An optional shuffling queue is created if shuffling_queue_capacity is greater than 0.

    Note that if reading a unischema field that is unicode (``np.unicode_`` or ``np.str_``) tensorflow will
    represent it as a tf.string which will be an array of bytes. If using python3 you may need to decode
    it to convert it back to a python str type.

    :param reader: An instance of petastorm.Reader object used as the data source
    :param shuffling_queue_capacity: Queue capacity is passed to the underlying :class:`tf.RandomShuffleQueue`
        instance. If set to 0, no suffling will be done.
    :param min_after_dequeue: If ``shuffling_queue_capacity > 0``, this value is passed to the underlying
        :class:`tf.RandomShuffleQueue`.
    :return: If no ngram reading is used, the function will return a named tuple with tensors that are populated
        from the underlying dataset. If ngram reading is enabled, a dictionary of named tuples of tensors is returned.
        The dictionary is indexed by time.
    """

    # NGram enabled and disabled code is quite different. It appears to be cleaner to simply go in orthogonal
    # execution paths.

    if reader.batched_output:
        if shuffling_queue_capacity > 0:
            raise ValueError('shuffling_queue_capacity can not be used with a reader that produces '
                             'batched_output, since each batch is a parquet read rowgroup. Extra '
                             'shuffling of the batches does not further decrease correlation.')

    if reader.ngram:
        result = _tf_tensors_ngram(reader, shuffling_queue_capacity, min_after_dequeue)
    else:
        result = _tf_tensors_nonngram(reader, shuffling_queue_capacity, min_after_dequeue)

    return result


def _set_shape_to_named_tuple(schema, fields, batched_output):
    """Assign static shape for all tensors"""
    fields_as_dict = fields._asdict()
    _set_shape(schema, fields_as_dict, batched_output)
    return schema.make_namedtuple_tf(**fields_as_dict)


def make_petastorm_dataset(reader):
    """Creates a `tensorflow.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_ object from
    a Petastorm :class:`~petastorm.reader.Reader`.

    The returned object can be used as any ``tf.data.Dataset`` with some limitations described below.

    * ``repeat``: An error will be raised if you call ``repeat`` on the returned dataset. Please use ``num_epochs``
      argument of the :meth:`~petastorm.reader.Reader` constructor.
    * ``shard``: Consider using ``training_partition`` and ``num_training_partitions`` arguments of the
      :class:`~petastorm.reader.Reader` constructor as it will not load any unused shards.
    * ``filter``: Consider using :class:`~petastorm.reader.Reader` ``predicate`` constructor argument.
      It will make use of columnar nature of the underlying Apache Parquet store to load only the columns that the
      predicate operates on prior to loading and decoding other columns. :class:`~petastorm.reader.Reader`'s predicate
      feature will also make use of Parquet partitioning (if the dataset is partitioned).

    The elements produced by the returned dataset object are namedtuples based on the
    :class:`~petastorm.unischema.Unischema`.

    >>> import tensorflow as tf
    >>> from petastorm.reader import Reader
    >>> from petastorm.tf_utils import make_petastorm_dataset
    >>>
    >>> with Reader('file:///some/path') as reader:
    >>>     dataset = make_petastorm_dataset(reader)
    >>>     next_sample = dataset.make_one_shot_iterator().get_next()
    >>>     with tf.Session() as sess:
    >>>         x = sess.run(next_sample)


    NGrams are not yet supported by this function.

    :param reader: An instance of :class:`~petastorm.reader.Reader` object that would serve as a data source.
    :return: A ``tf.data.Dataset`` instance.
    """

    if not reader.ngram:

        def dequeue_sample_impl():
            if reader.last_row_consumed:
                # This means that Dataset is trying to create a new instance of the generator. Can not do that
                # (nor want to do that) since this is an expensive operation. num_epochs is a more efficient way
                # to do this.
                raise RuntimeError('Multiple iterations over make_petastorm_dataset are not supported. '
                                   'Multiple iterations can be triggered by calling \'repeat\' method of Datset class.'
                                   'Use Reader\'s num_epochs contructor arguments to set number of iterations.')
            for row in reader:
                yield _sanitize_field_tf_types(row)

        flat_dataset = tf.data.Dataset.from_generator(dequeue_sample_impl, tuple(_schema_to_tf_dtypes(reader.schema)))
        named_tuple_dataset = flat_dataset \
            .map(reader.schema.make_namedtuple_tf) \
            .map(lambda row: _set_shape_to_named_tuple(reader.schema, row, reader.batched_output))
        return named_tuple_dataset
    else:
        raise NotImplementedError('make_petastorm_dataset does not support NGram yet.')
