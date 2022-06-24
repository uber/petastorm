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
import logging
import pickle
import sys
import os
from time import sleep, time
from traceback import format_exc

from threading import Thread
from psutil import pid_exists

import zmq
from zmq import ZMQBaseError

from petastorm.reader_impl.pickle_serializer import PickleSerializer
from petastorm.workers_pool import EmptyResultError, VentilatedItemProcessedMessage
from petastorm.workers_pool.exec_in_new_process import exec_in_new_process

# When _CONTROL_FINISHED is passed via control socket to a worker, the worker will terminate
_CONTROL_FINISHED = "FINISHED"
# This is the amount of seconds we will wait to all processes to be created. We throw an error if can not start them
# on time
_WORKERS_STARTED_TIMEOUT_S = 20
_SOCKET_LINGER_MS = 1000
_KEEP_TRYING_WHILE_ZMQ_AGAIN_IS_RAIZED_TIMEOUT_S = 20

# Amount of time we will wait on a the queue to get the next result. If no results received until then, we will
# recheck if no more items are expected to be ventilated
_VERIFY_END_OF_VENTILATION_PERIOD = 0.1

_WORKER_STARTED_INDICATOR = 'worker started indicator'

logger = logging.getLogger(__name__)


#
# ----------------                                    ------------------
# |              |  --- _ventilator_send  (push) -->  |                |
# | main process |  --- _control_sender   (pub)  -->  | worker process |
# |              |  <-- _results_receiver (pull)  --  |                |
# ----------------                                    ------------------
#
# 1. When ProcessPool start is called, it creates _ventilator_send, _control_sender and _result_receiver
#    sockets.
# 2. After initialization is done, worker process sends _WORKER_STARTED_INDICATOR
# 3. Once ProcessPool receives _WORKER_STARTED_INDICATOR from all workers, the ProcessPool
#    is ready to start ventilating.
#
# 4. Each ventilated message is picked up by one of the workers.
# 5. Worker process would send 0..n responses for each ventilated message. Each response
#    is a tuple of (data payload, control). Data payload is serialized using
#    _serializer instance. Control is always pickled.
# 6. After the last response to a single ventilated item is transmitted, an instance of VentilatedItemProcessedMessage
#    is transmitted as a control. This control message is needed to count how many ventilated
#    items are being processed at each time.
#
# 7. Workers are terminated by broadcasting _CONTROL_FINISHED message.
#


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
            return func()
        except zmq.Again:
            logger.debug('zmq.Again exception caught. Will try again')
            sleep(0.1)
            continue
        except ZMQBaseError as e:
            # There are race conditions while setting up the zmq socket so you can get unexpected errors
            # for the first bit of time. We therefore allow for a few unknown failures while the sockets
            # are warming up. Before propogating them as a true problem.
            sleep(0.1)
            failures += 1
            logger.debug('Unexpected ZMQ error \'%s\' received. Failures %d/%d', str(e), failures, allowed_failures)
            if failures > allowed_failures:
                raise
    raise RuntimeError('Timeout ({} [sec]) has elapsed while keep getting \'zmq.Again\''.format(timeout))


class ProcessPool(object):
    def __init__(self, workers_count, serializer=None, zmq_copy_buffers=True):
        """Initializes a ProcessPool.

        This pool is different from standard Python pool implementations by the fact that the workers are spawned
        without using fork. Some issues with using jvm based HDFS driver were observed when the process was forked
        (could not access HDFS from the forked worker if the driver was already used in the parent process).

        :param workers_count: Number of processes to be spawned
        :param serializer: An object that would be used for data payload serialization when sending data from a worker
          process to the main process. ``PickleSerializer`` is used by default. May use
          :class:`petastorm.reader_impl.ArrowTableSerializer` (should be used together with
          :class:`petastorm.reader.ArrowReader`)
        :param zmq_copy_buffers: When set to False, we will use a zero-memory-copy feature of recv_multipart.
          A downside of using this zero memory copy feature is that it does not play nice with Python GC and cases
          were observed when it resulted in wild memory footprint swings. Having the buffers copied is typically a
          safer alternative.
        """
        self._workers = []
        self._ventilator_send = None
        self._control_sender = None
        self.workers_count = workers_count
        self._results_receiver_poller = None
        self._results_receiver = None

        self._ventilated_items = 0
        self._ventilated_items_processed = 0
        self._ventilator = None
        self._serializer = serializer or PickleSerializer()
        self._zmq_copy_buffers = zmq_copy_buffers

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

        # Start a bunch of processes
        self._workers = [
            exec_in_new_process(_worker_bootstrap, worker_class, worker_id, control_socket, worker_receiver_socket,
                                results_sender_socket, os.getpid(), self._serializer, worker_setup_args)
            for worker_id in range(self.workers_count)]

        # Block until we have get a _WORKER_STARTED_INDICATOR from all our workers
        self._wait_for_workers_to_start()

        if ventilator:
            self._ventilator = ventilator
            self._ventilator.start()

    def _wait_for_workers_to_start(self):
        """Waits for all workers to start."""
        for _ in range(self.workers_count):
            started_indicator = _keep_retrying_while_zmq_again(
                _KEEP_TRYING_WHILE_ZMQ_AGAIN_IS_RAIZED_TIMEOUT_S,
                lambda: self._results_receiver.recv_pyobj(flags=zmq.constants.NOBLOCK))
            assert _WORKER_STARTED_INDICATOR == started_indicator

    def ventilate(self, *args, **kargs):
        """Sends a work item to a worker process. Will result in worker.process(...) call with arbitrary arguments."""
        self._ventilated_items += 1
        logger.debug('ventilate called. total ventilated items count %d', self._ventilated_items)
        # There is a race condition when sending objects to zmq that if all workers have been killed, sending objects
        # can block indefinitely. By using NOBLOCK, an exception is thrown stating that all resources have been
        # exhausted which the user can decide how to handle instead of just having the process hang.
        _keep_retrying_while_zmq_again(_KEEP_TRYING_WHILE_ZMQ_AGAIN_IS_RAIZED_TIMEOUT_S,
                                       lambda: self._ventilator_send.send_pyobj((args, kargs),
                                                                                flags=zmq.constants.NOBLOCK))

    def get_results(self):
        """Returns results from worker pool

        :param timeout: If None, will block forever, otherwise will raise :class:`.TimeoutWaitingForResultError`
            exception if no data received within the timeout (in seconds)
        :return: arguments passed to ``publish_func(...)`` by a worker. If no more results are anticipated,
            :class:`.EmptyResultError` is raised.
        """

        while True:
            # If there is no more work to do, raise an EmptyResultError
            logger.debug('ventilated_items=%d ventilated_items_processed=%d ventilator.completed=%s',
                         self._ventilated_items, self._ventilated_items_processed,
                         str(self._ventilator.completed()) if self._ventilator else 'N/A')
            if self._ventilated_items == self._ventilated_items_processed:
                # We also need to check if we are using a ventilator and if it is completed
                if not self._ventilator or self._ventilator.completed():
                    logger.debug('ventilator reported it has completed. Reporting end of results')
                    raise EmptyResultError()

            logger.debug('get_results polling on the next result')
            socks = self._results_receiver_poller.poll(_VERIFY_END_OF_VENTILATION_PERIOD * 1e3)
            if not socks:
                continue
            # Result message is a tuple containing data payload and possible exception (or None).
            fast_serialized, pickle_serialized = self._results_receiver.recv_multipart(copy=self._zmq_copy_buffers)
            pickle_serialized = pickle.loads(pickle_serialized)

            if pickle_serialized:
                logger.debug('get_results a pickled message %s', type(pickle_serialized))
                if isinstance(pickle_serialized, VentilatedItemProcessedMessage):
                    self._ventilated_items_processed += 1
                    if self._ventilator:
                        self._ventilator.processed_item()
                elif isinstance(pickle_serialized, Exception):
                    self.stop()
                    self.join()
                    raise pickle_serialized
            else:
                logger.debug('get_results received new results')
                if self._zmq_copy_buffers:
                    deserialized_result = self._serializer.deserialize(fast_serialized)
                else:
                    deserialized_result = self._serializer.deserialize(fast_serialized.buffer)
                return deserialized_result

    def stop(self):
        """Stops all workers (non-blocking)"""
        logger.debug('stopping')
        if self._ventilator:
            self._ventilator.stop()
        try:
            self._control_sender.send_string(_CONTROL_FINISHED)
        except ZMQBaseError as e:
            logger.warning('Stopping worker processes failed with \'%s\'. Does not necessary indicates an error.'
                           'This can happen if worker processes were terminated due to an error raised in that '
                           'process. See the log for additional messages from the failed worker.', str(e))

    def join(self):
        """Blocks until all workers are terminated."""

        logger.debug('joining')

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

    @property
    def diagnostics(self):
        # items_produced is updated only when VentilatedItemProcessedMessage is received. This will happen only on the
        # next call to get_results, so it's value may lag.
        return {
            'items_consumed': self._ventilated_items,
            'items_produced': self._ventilated_items_processed,
            'items_inprocess': self._ventilated_items - self._ventilated_items_processed,
            'zmq_copy_buffers': self._zmq_copy_buffers
        }


def _serialize_result_and_send(socket, serializer, data):
    # Result message is a tuple containing data payload and possible exception (or None).
    socket.send_multipart([serializer.serialize(data), pickle.dumps(None)])


def _monitor_thread_function(main_process_pid):
    while True:
        logger.debug('Monitor thread monitoring pid: %d', main_process_pid)
        main_process_alive = pid_exists(main_process_pid)
        if not main_process_alive:
            logger.debug('Main process with pid %d is dead. Killing worker', main_process_pid)
            os._exit(0)
        sleep(1)


def _worker_bootstrap(worker_class, worker_id, control_socket, worker_receiver_socket, results_sender_socket,
                      main_process_pid, serializer, worker_args):
    """This is the root of the spawned worker processes.

    :param worker_class: A class with worker implementation.
    :param worker_id: An integer. Unique for each worker.
    :param control_socket: zmq socket used to control the worker (currently supports only :class:`zmq.FINISHED` signal)
    :param worker_receiver_socket: A zmq socket used to deliver tasks to the worker
    :param results_sender_socket: A zmq socket used to deliver the work products to the consumer
    :param serializer: A serializer object (with serialize/deserialize methods) or None.
    :param worker_args: Application specific parameter passed to worker constructor
    :return: ``None``
    """
    logger.debug('Starting _worker_bootstrap')
    context = zmq.Context()

    logger.debug('Connecting sockets')
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

    logger.debug('Setting up poller')
    # Set up a poller to multiplex the work receiver and control receiver channels
    poller = zmq.Poller()
    poller.register(work_receiver, zmq.POLLIN)
    poller.register(control_receiver, zmq.POLLIN)

    results_sender.send_pyobj(_WORKER_STARTED_INDICATOR)

    # Use this 'none_marker' as the first argument to send_multipart.
    none_marker = bytes()

    logger.debug('Instantiating a worker')
    # Instantiate a worker
    worker = worker_class(worker_id, lambda data: _serialize_result_and_send(results_sender, serializer, data),
                          worker_args)

    logger.debug('Starting monitor loop')
    thread = Thread(target=_monitor_thread_function, args=(main_process_pid,))
    thread.daemon = True
    thread.start()

    # Loop and accept messages from both channels, acting accordingly
    logger.debug('Entering worker loop')
    while True:
        logger.debug('Polling new message')
        socks = dict(poller.poll())

        # If the message came from work_receiver channel
        if socks.get(work_receiver) == zmq.POLLIN:
            try:
                args, kargs = work_receiver.recv_pyobj()
                logger.debug('Starting worker.process')
                worker.process(*args, **kargs)
                logger.debug('Finished worker.process')
                results_sender.send_multipart([none_marker, pickle.dumps(VentilatedItemProcessedMessage())])
                logger.debug('Sending result')
            except Exception as e:  # pylint: disable=broad-except
                stderr_message = 'Worker %d terminated: unexpected exception:\n' % worker_id
                stderr_message += format_exc()
                logger.debug('worker.process failed with exception %s', stderr_message)
                sys.stderr.write(stderr_message)
                results_sender.send_multipart([none_marker, pickle.dumps(e)])
                return

        # If the message came over the control channel, shut down the worker.
        if socks.get(control_receiver) == zmq.POLLIN:
            control_message = control_receiver.recv_string()
            logger.debug('Received control message %s', control_message)
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
