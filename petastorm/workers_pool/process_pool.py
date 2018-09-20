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

"""This pool is different from standard Python pool implementations by the fact that the workers are spawned
without using fork. Some issues with using jvm based HDFS driver were observed when the process was forked
(could not access HDFS from the forked worker if the driver was already used in the parent process)"""
import sys
from decimal import Decimal
from time import sleep, time
from traceback import format_exc

import pyarrow
import zmq
from zmq import ZMQBaseError
from zmq.utils import monitor

from petastorm.workers_pool import EmptyResultError, VentilatedItemProcessedMessage, \
    TimeoutWaitingForResultError
from petastorm.workers_pool.exec_in_new_process import exec_in_new_process

# When _CONTROL_FINISHED is passed via control socket to a worker, the worker will terminate
_CONTROL_FINISHED = "FINISHED"
# This is the amount of seconds we will wait to all processes to be created. We throw an error if can not start them
# on time
_WORKERS_STARTED_TIMEOUT_S = 20
_SOCKET_LINGER_MS = 1000
_KEEP_TRYING_WHILE_ZMQ_AGAIN_IS_RAIZED_TIMEOUT_S = 20


def _keep_retrying_while_zmq_again(timeout, func, allowed_failures=3):
    """Will keep executing func() as long as zmq.Again is being thrown.

    Usage example:

    >>> _keep_retrying_while_zmq_again(
    >>>   _KEEP_TRYING_WHILE_ZMQ_AGAIN_IS_RAIZED_TIMEOUT_S,
    >>>   lambda: self._ventilator_send.send_pyobj(
    >>>      (args, kargs),
    >>>      flags=zmq.constants.NOBLOCK))

    :param timeout: A :class:`RuntimeError` is raised if could not execute ``func()`` without getting a
        :class:`zmq.Again` within this timeout. The timeout is defined in seconds.
    :param func: The function will be executed (as ``func()``)
    :return: None
    """
    now = time()
    failures = 0
    while time() < now + timeout:
        try:
            func()
        except zmq.Again:
            sleep(0.1)
            continue
        except ZMQBaseError:
            # There are race conditions while setting up the zmq socket so you can get unexpected errors
            # for the first bit of time. We therefore allow for a few unknown failures while the sockets
            # are warming up. Before propogating them as a true problem.
            sleep(0.1)
            failures += 1
            if failures > allowed_failures:
                raise
        return
    raise RuntimeError('Timeout ({} [sec]) has elapsed while keep getting \'zmq.Again\''.format(timeout))


class ProcessPool(object):
    def __init__(self, workers_count, pyarrow_serialize=False):
        """Initializes a ProcessPool.

        This pool is different from standard Python pool implementations by the fact that the workers are spawned
        without using fork. Some issues with using jvm based HDFS driver were observed when the process was forked
        (could not access HDFS from the forked worker if the driver was already used in the parent process).

        :param workers_count: Number of processes to be spawned
        :param pyarrow_serialize: Use ``pyarrow.serialize`` serialization if True. ``pyarrow.serialize`` is much faster
          than pickling, but does not support ``Decimal`` data types, converts int64 into int32 (and probably modifies
          some other types). We can not use this serialization by default, but would allow to switch it on
          when a user knows what they are doing.
        """
        self._workers = []
        self._ventilator_send = None
        self._control_sender = None
        self._workers_count = workers_count
        self._results_receiver_poller = None

        self._ventilated_items = 0
        self._ventilated_items_processed = 0
        self._ventilator = None
        self._pyarrow_serialize = pyarrow_serialize

    def _create_local_socket_on_random_port(self, context, socket_type):
        """Creates a zmq socket on a random port.

        :param context: zmq context
        :param socket_type: zmq socket type
        :return: A tuple: ``(zmq_socket, endpoint_address)``
        """
        LOCALHOST = 'tcp://127.0.0.1'
        socket = context.socket(socket_type)

        # There are race conditions where the socket can close when messages are still trying to be sent by zmq.
        # This can end up causing zmq to block indefinitely when sending objects or shutting down. Having the socket
        # linger on close helps prevent this.
        socket.linger = _SOCKET_LINGER_MS

        port = socket.bind_to_random_port(LOCALHOST)
        return socket, '{}:{}'.format(LOCALHOST, port)

    def start(self, worker_class, worker_setup_args=None, ventilator=None):
        """Starts worker processes.

        Will block until all processes to subscribe to the worker queue (the messages are distributed by zmq on write
        so if only one, out of many, workers is up at the time of 'ventilation', the initial load won't be balanced
        between workers. If can not start the workers in timely fashion, will raise an exception.

        :param worker_class: A class of the worker class. The class will be instantiated in the worker process. The
            class must implement :class:`.WorkerBase` protocol.
        :param worker_setup_args: Argument that will be passed to 'args' property of the instantiated
            :class:`.WorkerBase`.
        :param ventilator: Optional ventilator to handle ventilating items to the process pool. Process pool needs
            to know about the ventilator to know if it has completed ventilating items.
        :return: ``None``
        """
        # Initialize a zeromq context
        self._context = zmq.Context()

        # Ventilator socket used to send out tasks to workers
        self._ventilator_send, worker_receiver_socket = self._create_local_socket_on_random_port(self._context,
                                                                                                 zmq.PUSH)

        # Control socket is used to signal termination of the pool
        self._control_sender, control_socket = self._create_local_socket_on_random_port(self._context, zmq.PUB)
        self._results_receiver, results_sender_socket = self._create_local_socket_on_random_port(self._context,
                                                                                                 zmq.PULL)

        # We need poller to be able to read results from workers in a non-blocking manner
        self._results_receiver_poller = zmq.Poller()
        self._results_receiver_poller.register(self._results_receiver, zmq.POLLIN)

        # Monitors will be used to count number of workers created.
        # We will block till all of them are ready to accept messages
        monitor_sockets = [
            self._ventilator_send.get_monitor_socket(zmq.constants.EVENT_ACCEPTED),
            self._control_sender.get_monitor_socket(zmq.constants.EVENT_ACCEPTED),
            self._results_receiver.get_monitor_socket(zmq.constants.EVENT_ACCEPTED),
        ]

        # Start a bunch of processes
        self._workers = [
            exec_in_new_process(_worker_bootstrap, worker_class, worker_id, control_socket, worker_receiver_socket,
                                results_sender_socket, self._pyarrow_serialize, worker_setup_args)
            for worker_id in range(self._workers_count)]

        # Block until we have all workers up. Will raise an error if fails to start in a timely fashion
        self._wait_for_workers_to_start(monitor_sockets)

        if ventilator:
            self._ventilator = ventilator
            self._ventilator.start()

    def _wait_for_workers_to_start(self, monitor_sockets):
        """Waits for all workers to start."""
        now = time()
        for monitor_socket in monitor_sockets:
            started_count = 0
            while started_count < self._workers_count and time() < now + _WORKERS_STARTED_TIMEOUT_S:
                _keep_retrying_while_zmq_again(_KEEP_TRYING_WHILE_ZMQ_AGAIN_IS_RAIZED_TIMEOUT_S,
                                               lambda sock=monitor_socket: monitor.recv_monitor_message(
                                                   sock, flags=zmq.constants.NOBLOCK))
                started_count += 1

            if started_count < self._workers_count:
                raise RuntimeError(
                    'Workers were not able to start within timeout {} s ({} has started)'.format(
                        _WORKERS_STARTED_TIMEOUT_S,
                        started_count))

    def ventilate(self, *args, **kargs):
        """Sends a work item to a worker process. Will result in worker.process(...) call with arbitrary arguments."""
        self._ventilated_items += 1

        # There is a race condition when sending objects to zmq that if all workers have been killed, sending objects
        # can block indefinitely. By using NOBLOCK, an exception is thrown stating that all resources have been
        # exhausted which the user can decide how to handle instead of just having the process hang.
        _keep_retrying_while_zmq_again(_KEEP_TRYING_WHILE_ZMQ_AGAIN_IS_RAIZED_TIMEOUT_S,
                                       lambda: self._ventilator_send.send_pyobj((args, kargs),
                                                                                flags=zmq.constants.NOBLOCK))

    def get_results(self, timeout=None):
        """Returns results from worker pool

        :param timeout: If None, will block forever, otherwise will raise :class:`.TimeoutWaitingForResultError`
            exception if no data received within the timeout (in seconds)
        :return: arguments passed to ``publish_func(...)`` by a worker. If no more results are anticipated,
            :class:`.EmptyResultError` is raised.
        """

        while True:
            # If there is no more work to do, raise an EmptyResultError
            if self._ventilated_items == self._ventilated_items_processed:
                # We also need to check if we are using a ventilator and if it is completed
                if not self._ventilator or self._ventilator.completed():
                    raise EmptyResultError()

            socks = self._results_receiver_poller.poll(timeout * 1e3 if timeout else None)
            if not socks:
                raise TimeoutWaitingForResultError()
            result = self._results_receiver.recv_pyobj(0)
            if isinstance(result, VentilatedItemProcessedMessage):
                self._ventilated_items_processed += 1
                if self._ventilator:
                    self._ventilator.processed_item()
                continue
            if isinstance(result, Exception):
                self.stop()
                self.join()
                raise result
            else:
                if self._pyarrow_serialize:
                    deserialized_result = pyarrow.read_serialized(result).deserialize()
                else:
                    deserialized_result = result

                return deserialized_result

    def stop(self):
        """Stops all workers (non-blocking)"""
        if self._ventilator:
            self._ventilator.stop()
        self._control_sender.send_string(_CONTROL_FINISHED)

    def join(self):
        """Blocks until all workers are terminated."""

        # Slow joiner problem with zeromq means that not all workers are guaranteed to have gotten
        # the stop event. Therefore we will keep sending it until all workers are stopped to prevent
        # a deadlock.
        while any([w.poll() is None for w in self._workers]):
            self.stop()
            sleep(.1)

        for w in self._workers:
            w.wait()
        self._ventilator_send.close()
        self._control_sender.close()
        self._results_receiver.close()
        self._context.destroy()


def _decimal_to_str(data):
    """Iterates over a nested structure of lists and dictionaries while substituting all Decimal instances with
    normalized string representation of Decimals.

    Modification is done in-place.

    :param data: A nested structure of lists and dictionaries
    :return: None
    """
    if isinstance(data, list):
        for row in data:
            _decimal_to_str(row)
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                _decimal_to_str(v)
            else:
                if isinstance(v, Decimal):
                    data[k] = str(v.normalize())


def _serialize_result_and_send(socket, pyarrow_serialize, data):
    if pyarrow_serialize:
        # pyarrow won't be able to serialize Decimals. Replace all Decimal with strings.
        _decimal_to_str(data)
        serialized = pyarrow.serialize(data)
        socket.send_pyobj(serialized.to_buffer())
    else:
        socket.send_pyobj(data)


def _worker_bootstrap(worker_class, worker_id, control_socket, worker_receiver_socket, results_sender_socket,
                      pyarrow_serialize, worker_args):
    """This is the root of the spawned worker processes.

    :param worker_class: A class with worker implementation.
    :param worker_id: An integer. Unique for each worker.
    :param control_socket: zmq socket used to control the worker (currently supports only :class:`zmq.FINISHED` signal)
    :param worker_receiver_socket: A zmq socket used to deliver tasks to the worker
    :param results_sender_socket: A zmq socket used to deliver the work products to the consumer
    :param worker_args: Application specific parameter passed to worker constructor
    :return: ``None``
    """
    context = zmq.Context()

    # Set up a channel to receive work from the ventilator
    work_receiver = context.socket(zmq.PULL)
    work_receiver.linger = 0
    work_receiver.connect(worker_receiver_socket)

    # Set up a channel to send result of work to the results reporter
    results_sender = context.socket(zmq.PUSH)
    results_sender.linger = 0
    results_sender.connect(results_sender_socket)

    # Set up a channel to receive control messages over
    control_receiver = context.socket(zmq.SUB)
    control_receiver.linger = 0
    control_receiver.connect(control_socket)
    _setsockopt(control_receiver, zmq.SUBSCRIBE, b"")

    # Set up a poller to multiplex the work receiver and control receiver channels
    poller = zmq.Poller()
    poller.register(work_receiver, zmq.POLLIN)
    poller.register(control_receiver, zmq.POLLIN)

    # Instantiate a worker
    worker = worker_class(worker_id, lambda data: _serialize_result_and_send(results_sender, pyarrow_serialize,
                                                                             data), worker_args)

    # Loop and accept messages from both channels, acting accordingly
    while True:
        socks = dict(poller.poll())

        # If the message came from work_receiver channel
        if socks.get(work_receiver) == zmq.POLLIN:
            try:
                args, kargs = work_receiver.recv_pyobj()
                worker.process(*args, **kargs)
                results_sender.send_pyobj(VentilatedItemProcessedMessage())
            except Exception as e:  # pylint: disable=broad-except
                stderr_message = 'Worker %d terminated: unexpected exception:\n' % worker_id
                stderr_message += format_exc()
                sys.stderr.write(stderr_message)
                results_sender.send_pyobj(e)
                return

        # If the message came over the control channel, shut down the worker.
        if socks.get(control_receiver) == zmq.POLLIN:
            control_message = control_receiver.recv_string()
            if control_message == _CONTROL_FINISHED:
                worker.shutdown()
                break


def _setsockopt(sock, option, value):
    """
    This wraps setting socket options since python2 vs python3 handles strings differently
    and pyzmq requires a different call. See http://pyzmq.readthedocs.io/en/latest/unicode.html
    """
    try:
        sock.setsockopt(option, value)
    except TypeError:
        sock.setsockopt_string(option, value)
