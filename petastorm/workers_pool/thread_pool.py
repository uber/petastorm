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

import cProfile
import pstats
import sys
from threading import Thread, Event
from traceback import format_exc

from six.moves import queue

from petastorm.workers_pool import EmptyResultError, VentilatedItemProcessedMessage

# Defines how frequently will we check the stop event while waiting on a blocking queue
IO_TIMEOUT_INTERVAL_S = 0.001
# Amount of time we will wait on a the queue to get the next result. If no results received until then, we will
# recheck if no more items are expected to be ventilated
_VERIFY_END_OF_VENTILATION_PERIOD = 0.1


class WorkerTerminationRequested(Exception):
    """This exception will be raised if a thread is being stopped while waiting to write to the results queue."""


class WorkerThread(Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, worker_impl, stop_event, ventilator_queue, results_queue, profiling_enabled=False):
        super(WorkerThread, self).__init__()
        self._stop_event = stop_event
        self._worker_impl = worker_impl
        self._ventilator_queue = ventilator_queue
        self._results_queue = results_queue
        self._profiling_enabled = profiling_enabled
        if profiling_enabled:
            self.prof = cProfile.Profile()

    def run(self):
        if self._profiling_enabled:
            self.prof.enable()
        # Loop and accept messages from both channels, acting accordingly
        while True:
            # Check for stop event first to prevent erroneous reuse
            if self._stop_event.is_set():
                break
            # If the message came from work_receiver channel
            try:
                (args, kargs) = self._ventilator_queue.get(block=True, timeout=IO_TIMEOUT_INTERVAL_S)
                self._worker_impl.process(*args, **kargs)
                self._worker_impl.publish_func(VentilatedItemProcessedMessage())
            except queue.Empty:
                pass
            except WorkerTerminationRequested:
                pass
            except Exception as e:  # pylint: disable=broad-except
                stderr_message = 'Worker %d terminated: unexpected exception:\n' % self._worker_impl.worker_id
                stderr_message += format_exc()
                sys.stderr.write(stderr_message)
                self._results_queue.put(e)
                break
        if self._profiling_enabled:
            self.prof.disable()


class ThreadPool(object):
    def __init__(self, workers_count, results_queue_size=50, profiling_enabled=False, shuffle_rows=False, seed=None):
        """Initializes a thread pool.

        TODO: consider using a standard thread pool
        (e.g. http://elliothallmark.com/2016/12/23/requests-with-concurrent-futures-in-python-2-7/ as an implementation)

        Originally implemented our own pool to match the interface of ProcessPool (could not find a process pool
        implementation that would not use fork)

        :param workers_count: Number of threads
        :param profile: Whether to run a profiler on the threads
        """
        self._seed = seed
        self._shuffle_rows = shuffle_rows
        self._workers = []
        self._ventilator_queues = []
        self.workers_count = workers_count
        self._results_queue_size = results_queue_size
        # Worker threads will watch this event and gracefully shutdown when the event is set
        self._stop_event = Event()
        self._profiling_enabled = profiling_enabled

        # Count of items ventilated by the pool
        self._ventilated_items = 0
        # Count of items ventilated by each worker
        self._ventilated_items_by_worker = [0 for _ in range(self.workers_count)]
        # Count of items processed by each worker
        self._ventilated_items_processed_by_worker = [0 for _ in range(self.workers_count)]
        self._ventilator = None
        self._get_results_worker_id = 0

    def start(self, worker_class, worker_args=None, ventilator=None):
        """Starts worker threads.

        :param worker_class: A class of the worker class. The class will be instantiated in the worker process. The
          class must implement :class:`.WorkerBase` protocol
        :param worker_setup_args: Argument that will be passed to ``args`` property of the instantiated
          :class:`.WorkerBase`
        :return: ``None``
        """
        # Verify stop_event and raise exception if it's already set!
        if self._stop_event.is_set():
            raise RuntimeError('ThreadPool({}) cannot be reused! stop_event set? {}'
                               .format(len(self._workers), self._stop_event.is_set()))

        # Set up a channel for each worker to send work
        self._ventilator_queues = [queue.Queue() for _ in range(self.workers_count)]
        # Set up a channel for each worker to send results
        self._results_queues = [queue.Queue(5) for _ in range(self.workers_count)]
        self._workers = []
        for worker_id in range(self.workers_count):
            # Create a closure that captures the worker_id for this specific worker
            def make_publish_func(worker_id):
                return lambda data: self._stop_aware_put(worker_id, data)

            worker_impl = worker_class(worker_id, make_publish_func(worker_id), worker_args)
            new_thread = WorkerThread(worker_impl, self._stop_event, self._ventilator_queues[worker_id],
                                      self._results_queues[worker_id], self._profiling_enabled)
            # Make the thread daemonic. Since it only reads it's ok to abort while running - no resource corruption
            # will occur.
            new_thread.daemon = True
            self._workers.append(new_thread)

        # Spin up all worker threads
        for w in self._workers:
            w.start()

        if ventilator:
            self._ventilator = ventilator
            self._ventilator.start()

    def ventilate(self, *args, **kargs):
        """Sends a work item to a worker process. Will result in ``worker.process(...)`` call with arbitrary arguments.
        """
        current_worker_id = self._ventilated_items % self.workers_count
        self._ventilated_items += 1
        self._ventilated_items_by_worker[current_worker_id] += 1
        self._ventilator_queues[current_worker_id].put((args, kargs))

    def current_worker_done(self, worker_id):
        # Check if the current worker has processed all the items it was assigned and if the results queue is empty
        return (self._ventilated_items_processed_by_worker[worker_id] == self._ventilated_items_by_worker[worker_id]
                and self._results_queues[worker_id].empty())

    def all_workers_done(self):
        # Check if all workers have processed all the items they were assigned and if the results queues are empty
        for i in range(self.workers_count):
            if not self.current_worker_done(i):
                return False
        return True

    def get_results(self):
        """Returns results from worker pool or re-raise worker's exception if any happen in worker thread.

        :param timeout: If None, will block forever, otherwise will raise :class:`.TimeoutWaitingForResultError`
            exception if no data received within the timeout (in seconds)

        :return: arguments passed to ``publish_func(...)`` by a worker. If no more results are anticipated,
                 :class:`.EmptyResultError`.
        """
        # If shuffle_rows is enabled and the seed is not set, we need to use a non-blocking
        #  as we don't care about the strict round robin order
        use_non_blocking_get = self._shuffle_rows and (self._seed is None or self._seed == 0)
        while True:
            # If there is no more work to do, raise an EmptyResultError
            if self.all_workers_done():
                # We also need to check if we are using a ventilator and if it is completed
                if not self._ventilator or self._ventilator.completed():
                    raise EmptyResultError()

            # If the current worker is done, we need to get the result from the next worker
            if self.current_worker_done(self._get_results_worker_id):
                self._get_results_worker_id = (self._get_results_worker_id + 1) % self.workers_count
                continue

            try:
                # Get the result from the current worker's results queue.
                # Use blocking/strict round robin if shuffle_rows is disabled or the seed is set
                result = self._results_queues[self._get_results_worker_id].get(
                    block=not use_non_blocking_get, timeout=_VERIFY_END_OF_VENTILATION_PERIOD
                )
                # If the result is a VentilatedItemProcessedMessage, we need to increment the count of items
                # processed by the current worker
                if isinstance(result, VentilatedItemProcessedMessage):
                    self._ventilated_items_processed_by_worker[self._get_results_worker_id] += 1
                    if self._ventilator:
                        self._ventilator.processed_item()
                    # Move to the next worker
                    self._get_results_worker_id = (self._get_results_worker_id + 1) % self.workers_count
                    continue
                elif isinstance(result, Exception):
                    self.stop()
                    self.join()
                    raise result
                else:
                    return result
            except queue.Empty:
                continue

    def stop(self):
        """Stops all workers (non-blocking)."""
        if self._ventilator:
            self._ventilator.stop()
        self._stop_event.set()

    def join(self):
        """Block until all workers are terminated."""
        for w in self._workers:
            if w.is_alive():
                w.join()

        if self._profiling_enabled:
            # If we have profiling set, collect stats and print them
            stats = None
            for w in self._workers:
                if stats:
                    stats.add(w.prof)
                else:
                    stats = pstats.Stats(w.prof)
            stats.sort_stats('cumulative').print_stats()

    def _stop_aware_put(self, worker_id, data):
        """This method is called to write the results to the results queue. We use ``put`` in a non-blocking way so we
        can gracefully terminate the worker thread without being stuck on :func:`Queue.put`.

        The method raises :class:`.WorkerTerminationRequested` exception that should be passed through all the way up to
        :func:`WorkerThread.run` which will gracefully terminate main worker loop."""
        while True:
            try:
                self._results_queues[worker_id].put(data, block=True, timeout=IO_TIMEOUT_INTERVAL_S)
                return
            except queue.Full:
                pass

            if self._stop_event.is_set():
                raise WorkerTerminationRequested()

    def results_qsize(self):
        return sum(queue.qsize() for queue in self._results_queues)

    @property
    def diagnostics(self):
        return {'output_queue_size': self.results_qsize()}
