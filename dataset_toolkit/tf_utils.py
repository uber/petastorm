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
    np.float32: tf.float32,
    np.float64: tf.float64,
    np.string_: tf.string,
    np.bool_: tf.bool,
    Decimal: tf.string,
}

# Name of an op in the TF graph used for the random shuffling queue. This name can be used by diagnostics code that
# wishes to read-out shuffling queue size
RANDOM_SHUFFLING_QUEUE_SIZE = 'random_shuffling_queue_size'


def _char0bug_workaround(string_tensor):
    # CAUTION: this would unintentionally modify a string if this is a binary string that have null characters. A real
    # solution would be to fix tensorflow bug causing this in the first place!
    return tf.string_split(string_tensor, delimiter='\x00', skip_empty=True).values


def _sanitize_field_tf_types(sample):
    """Takes a named tuple and casts/promotes types unknown to TF to the types that are known.

    Two casts that are currently implemented
      - Decimal to string
      - uint16 to int32

    :param sample: named tuple or a dictoinary
    :return: same type as the input with values casted to types supported by Tensorflow
    """
    next_sample_dict = sample._asdict()

    for k, v in next_sample_dict.iteritems():
        if v is None:
            raise RuntimeError('Encountered "{}"=None. Tensorflow does not support None values as a tensor.'
                               'Consider filtering out these rows using a predicate.'.format(k))
        # Assuming conversion to the same numpy type is trivial and dirty cheap
        if isinstance(v, Decimal):
            # Normalizing decimals only to get rid of the trailing zeros (makes testing easier, assuming has
            # no other effect)
            next_sample_dict[k] = str(v.normalize())
        elif isinstance(v, np.ndarray) and v.dtype == np.uint16:
            next_sample_dict[k] = v.astype(np.int32)

    # Construct object of the same type as the input
    return sample.__class__(**next_sample_dict)


def _schema_to_tf_dtypes(schema):
    """
    Returns schema as a list of tensorflow dtypes.
    :param schema: The schema.
    :return: List of tensorflow dtypes.
    """
    return [_numpy_to_tf_dtypes(f.numpy_dtype) for f in schema.fields.values()]


def _schema_to_tf_dtypes_sequence(schema, sequence):
    """
    Returns schema as a list of tensorflow dtypes for a sequence.
    :param schema: The schema.
    :param sequence: The sequence.
    :return: tensorflow dtypes for a sequence.
    """
    result = []
    # Iterate over each timestep
    for key in range(sequence.length):
        # Iterate over each field
        for field in schema.fields.values():
            result.append(_numpy_to_tf_dtypes(field.numpy_dtype))
    return result


def _numpy_to_tf_dtypes(numpy_dtype):
    """Returns a tensorflow dtype object corresponding to numpy's dtype.

    A ValueError is raised if there is no known mapping between the types

    :param numpy_dtype: numpy dtype object
    :return: tensorflow dtype object
    """
    if numpy_dtype in _NUMPY_TO_TF_DTYPES_MAPPING:
        return _NUMPY_TO_TF_DTYPES_MAPPING[numpy_dtype]
    else:
        raise ValueError('Unknown mapping of numpy {} to tensorflow dtype'.format(numpy_dtype))


def _flatten(data):
    """
    Flattens the data, where it takes a dictionary of timesteps, each value is a dictionary and converts it to
    one flat dictionary having a key that is the key of the inner dictionary + '_' + timestep.

    For example, data would be {1: {'a': 'avalue', 'b': 'bvalue'}, 2: {'c': 'cvalue', 'd': 'dvalue'}} and the
    output of _flatten would be {'a_1': 'avalue', 'b_1': 'bvalue', 'c_2': 'cvalue', 'd_2': 'dvalue'}.

    :param data: The data to flatten.
    :return: The flattened dictionary.
    """
    flattened = OrderedDict()
    for key in data:
        data_dict = data[key]._asdict()
        for subkey in data_dict:
            encoded_key = subkey + '_' + str(key)
            flattened[encoded_key] = data_dict[subkey]

    FlattenedTuple = namedtuple('flattened', flattened.keys())
    return FlattenedTuple(**flattened)


def make_namedtuple_tf_sequence(unischema, sequence, *args, **kargs):
    """
    Creates a dictionary of timestep keys and namedtuple values from args and kargs.

    :param sequence: The sequence definition.
    :param args: args.
    :param kargs: kargs
    :return: A dictionary of timestep keys and namedtuple values.
    """
    sequence_result = {}
    for timestep in range(sequence.length):
        # For each timestep iteration, mark the args and kargs for that timestep and create
        # a namedtuple from them.
        fields_length = len(unischema._fields)
        args_timestep = args[timestep * fields_length:(timestep + 1) * fields_length]
        kargs_timestep = (kargs[str(timestep)] if str(timestep) in kargs else {})
        sequence_result[timestep] = unischema._get_namedtuple()(*args_timestep, **kargs_timestep)
    return sequence_result


def _set_shape(schema, fields_as_dict):
    # Assign static shape for all tensors
    # Workaround of an issue described here:
    # https://stackoverflow.com/questions/49161316/trailing-x00-characters-in-tensor-when-numpy-string-array-is-returned-from-tf
    for k in fields_as_dict.keys():
        unischema_field = schema.fields[k]

        # Set static shape
        fields_as_dict[k].set_shape(unischema_field.shape)

        # Workaround trailing null characters
        if unischema_field.numpy_dtype == np.string_ and len(fields_as_dict[k].get_shape()) == 1:
            fields_as_dict[k] = _char0bug_workaround(fields_as_dict[k])


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


def _tf_tensors_nonsequence(reader, shuffling_queue_capacity, min_after_dequeue):
    """A tensorflow data adapter for non sequences. Return value is a named tuple with tensorflow tensors supplying
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
    _set_shape(reader.schema, fields_as_dict)

    # Make a row tensor into a nice named tuple
    return reader.schema.make_namedtuple_tf(**fields_as_dict)


def _tf_tensors_sequence(reader, shuffling_queue_capacity, min_after_dequeue):
    """A tensorflow data adapter for sequences. Return value is a named tuple with tensorflow tensors supplying
    the data directly into a Tensoflow graph. See `tf_tensor` documentation for input/output arguments meaning."""

    # TODO(yevgeni): implement a mechanism for signaling that we have no more data
    def dequeue_sample_impl(x):
        next_sample = next(reader)
        assert (isinstance(next_sample, dict))

        # Create a dictionary, where each key is a timestep, and value is named tuple or dictionary.
        sequence = {}
        for timestep in next_sample:
            sequence[timestep] = _sanitize_field_tf_types(next_sample[timestep])

        return _flatten(sequence)

    fields_as_list = tf.py_func(dequeue_sample_impl, [tf.constant(1)],
                                _schema_to_tf_dtypes_sequence(reader.schema, reader.sequence))

    if shuffling_queue_capacity > 0:
        # Pass py_func output via shuffling queue if requested.
        fields_as_list = _shuffling_queue(shuffling_queue_capacity, min_after_dequeue,
                                          _schema_to_tf_dtypes_sequence(reader.schema, reader.sequence), fields_as_list)

    fields_as_namedtuple = make_namedtuple_tf_sequence(reader.schema, reader.sequence, *fields_as_list)

    # We change the key to str format here in order to be able to use ** later to expand the dictionary as kargs.
    fields_as_dict = {
        str(timestep): fields_as_namedtuple[timestep]._asdict() for timestep in fields_as_namedtuple}
    for timestep in fields_as_dict:
        _set_shape(reader.schema, fields_as_dict[timestep])

    return make_namedtuple_tf_sequence(reader.schema, reader.sequence, **fields_as_dict)


def tf_tensors(reader, shuffling_queue_capacity=0, min_after_dequeue=0):
    """Bridges between python-only interface of the Reader (next(Reader)) and tensorflow world.

    This function returns a named tuple of tensors form the dataset, e.g.
    >>> row_tensors
    >>> Out[2]: TestSchema_view(field_1=<tf.Tensor 'PyFunc:0' shape=() dtype=string>,
    >>>         field_2=<tf.Tensor 'StringSplit:1' shape=(?,) dtype=string>,
    >>>         field_3=<tf.Tensor 'PyFunc:2' shape=() dtype=int64>, ...)

    If the reader was created with `sequence=Sequence(...)` parameter, then a dictionary of named tuples is returned
    (indexed by time)
    >>> row_tensors
    >>> Out[6]:
    >>> {0: TestSchema_view(field_1=<tf.Tensor 'PyFunc_4:0' shape=() dtype=string>, field_2=...),
    >>>  1: TestSchema_view(field_1=<tf.Tensor 'PyFunc_4:11' shape=() dtype=string>, field_2=...),
    >>>  2: TestSchema_view(field_1=<tf.Tensor 'PyFunc_4:22' shape=() dtype=string>, field_2=...)}

    An optional shuffling queue is created if shuffling_queue_capacity is greater than 0

    :param reader: An instance of dataset_toolkit.Reader object used as the data source
    :param shuffling_queue_capacity: Queue capacity is passed to the underlying tf.RandomShuffleQueue instance. If set
    to 0, no suffling will be done.
    :param min_after_dequeue: If shuffling_queue_capacity>0, this value is passed to the underlying
    tf.RandomShuffleQueue

    :return: If no sequence reading is used, the function will return a named tuple with tensors that are populated
    from the underlying dataset. If sequence reading is enabled, a dictionary of named tuples of tensors is returned.
    The dictionary is indexed by time.
    """

    # Sequence enabled and disabled code is quite different. It appears to be cleaner to simply go in orthogonal
    # execution paths.
    if reader.sequence:
        result = _tf_tensors_sequence(reader, shuffling_queue_capacity, min_after_dequeue)
    else:
        result = _tf_tensors_nonsequence(reader, shuffling_queue_capacity, min_after_dequeue)

    return result
