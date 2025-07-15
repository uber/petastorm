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
import logging
import pstats
import random
import sys
import os
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
        self._items_processed = 0
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
                item = self._ventilator_queue.get(block=True, timeout=IO_TIMEOUT_INTERVAL_S)
                self._worker_impl.process(**item)
                self._worker_impl.publish_func(VentilatedItemProcessedMessage())
                self._items_processed += 1  # Only increment for actual data items
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
    
    def is_worker_done(self):
        return self._items_processed == self._worker_impl.thread_pool._items_per_worker[self._worker_impl.worker_id]


class ThreadPool(object):
    def __init__(self, workers_count, results_queue_size=25, worker_results_queue_size=5, profiling_enabled=False, shuffle_rows=False, seed=None):
        """Initializes a thread pool.

        TODO: consider using a standard thread pool
        (e.g. http://elliothallmark.com/2016/12/23/requests-with-concurrent-futures-in-python-2-7/ as an implementation)

        Originally implemented our own pool to match the interface of ProcessPool (could not find a process pool
        implementation that would not use fork)

        :param workers_count: Number of threads
        :param profiling_enabled: Whether to run a profiler on the threads
        :param shuffle_rows: Whether to shuffle rows (affects round-robin behavior)
        :param seed: Random seed for deterministic behavior
        """

        self._workers = []
        self._ventilator_queues = None
        self.workers_count = workers_count
        self._results_queue_size = results_queue_size
        self._worker_results_queue_size = worker_results_queue_size
        # Worker threads will watch this event and gracefully shutdown when the event is set
        self._stop_event = Event()
        self._profiling_enabled = profiling_enabled
        self._shuffle_rows = shuffle_rows
        self._seed = seed

        self._ventilated_items = 0
        self._ventilated_items_processed = 0
        self._ventilator = None
        
        # Round-robin consumer thread
        self._round_robin_thread = None
        self._items_per_worker = [0] * workers_count
        
  
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
            error_msg = f'ThreadPool({len(self._workers)}) cannot be reused! stop_event set? {self._stop_event.is_set()}'
            raise RuntimeError(error_msg)

        
        # Set up a channel to send work
        self._ventilator_queues = [queue.Queue() for _ in range(self.workers_count)]
        self._results_queues = [queue.Queue(self._worker_results_queue_size) for _ in range(self.workers_count)]
        self._shared_results_queue = queue.Queue(self._results_queue_size)
        self._workers = []
        self.thread_pool = self
        
        
        for worker_id in range(self.workers_count):
            # Create a closure that captures the worker_id for this specific worker
            def make_publish_func(worker_id):
                return lambda data: self._stop_aware_put(data, worker_id)
            
            worker_impl = worker_class(worker_id, make_publish_func(worker_id), worker_args)
            # Add thread_pool reference to worker for status tracking
            worker_impl.thread_pool = self
            
            new_thread = WorkerThread(worker_impl, self._stop_event, self._ventilator_queues[worker_id],
                                      self._results_queues[worker_id], self._profiling_enabled)
            # Make the thread daemonic. Since it only reads it's ok to abort while running - no resource corruption
            # will occur.
            new_thread.daemon = True
            self._workers.append(new_thread)
            

        # Spin up all worker threads
        for i, w in enumerate(self._workers):
            w.start()

        self._round_robin_thread = Thread(target=self._round_robin_consumer, daemon=True)
        
        if ventilator:
            self._ventilator = ventilator
            self._ventilator.start()

    def ventilate(self, items_to_ventilate):
        """Sends a work item to a worker process. Will result in ``worker.process(...)`` call with arbitrary arguments.
        """

        for i, item in enumerate(items_to_ventilate):
            worker_id = i % self.workers_count
            self._ventilator_queues[worker_id].put(item)
            self._items_per_worker[worker_id] += 1
            self._ventilated_items += 1
        
        # Start the round-robin consumer after ventilation has started
        if self._round_robin_thread and not self._round_robin_thread.is_alive():
            self._round_robin_thread.start()
         

    def all_workers_done(self):
        for worker_id in range(self.workers_count):
            if not self._results_queues[worker_id].empty() or not self._ventilator_queues[worker_id].empty() or not self._workers[worker_id].is_worker_done():
                return False
        return True
    
    def completed(self):
        # If all workers are done and shared queue is empty, raise EmptyResultError
        if self.all_workers_done() and self._shared_results_queue.empty():
            if not self._ventilator or self._ventilator.completed():
                return True
        return False


    def get_results(self):
        """Returns results from worker pool or re-raise worker's exception if any happen in worker thread.

        :param timeout: If None, will block forever, otherwise will raise :class:`.TimeoutWaitingForResultError`
            exception if no data received within the timeout (in seconds)

        :return: arguments passed to ``publish_func(...)`` by a worker. If no more results are anticipated,
                 :class:`.EmptyResultError`.
        """

        while True:
            # Check termination condition: all workers are truly done and shared queue is empty
            if self.completed():
                raise EmptyResultError()

            try:

                result = self._shared_results_queue.get(timeout=_VERIFY_END_OF_VENTILATION_PERIOD)
                if isinstance(result, Exception):
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
        
        for i, w in enumerate(self._workers):
            if w.is_alive():
                w.join()

        # Join the round-robin consumer thread
        if self._round_robin_thread and self._round_robin_thread.is_alive():
            self._round_robin_thread.join()

        if self._profiling_enabled:
            stats = None
            for w in self._workers:
                if stats:
                    stats.add(w.prof)
                else:
                    stats = pstats.Stats(w.prof)
            stats.sort_stats('cumulative').print_stats()
        
        if self._profiling_enabled:
            stats = None
            for w in self._workers:
                if stats:
                    stats.add(w.prof)
                else:
                    stats = pstats.Stats(w.prof)
            stats.sort_stats('cumulative').print_stats()


    def _stop_aware_put(self, data, worker_id):
        """This method is called to write the results to the results queue. We use ``put`` in a non-blocking way so we
        can gracefully terminate the worker thread without being stuck on :func:`Queue.put`.

        The method raises :class:`.WorkerTerminationRequested` exception that should be passed through all the way up to
        :func:`WorkerThread.run` which will gracefully terminate main worker loop."""
        
        # Skip control messages - they shouldn't go into the results queue
        if isinstance(data, VentilatedItemProcessedMessage):
            return
            
        while True:
            try:
                self._results_queues[worker_id].put(data, block=True, timeout=IO_TIMEOUT_INTERVAL_S)
                return
            except queue.Full:
                pass

            if self._stop_event.is_set():
                raise WorkerTerminationRequested()

    def _round_robin_consumer(self):
        """Round-robin consumer that takes items from each worker's queue in strict round-robin order
        and puts them into the shared results queue."""
        
        current_worker = 0
        
        # Determine if we should use non-blocking behavior
        use_non_blocking = self._shuffle_rows and (self._seed is None or self._seed == 0)
        
        while not self._stop_event.is_set():
            try:
                if self.all_workers_done() :
                    break
                # Check if current worker should be skipped
                should_skip = (
                    self._results_queues[current_worker].empty() and  # Worker result queue is empty
                    self._workers[current_worker].is_worker_done() and          # Worker is done
                    self._ventilator_queues[current_worker].empty()   # Ventilator queue for worker is empty
                )
                
                if should_skip:
                    # Skip this worker and move to next
                    current_worker = (current_worker + 1) % self.workers_count
                    continue
                
                # Try to get an item from the current worker's queue
                if use_non_blocking:
                    # Non-blocking: try to get item without waiting
                    try:
                        item = self._results_queues[current_worker].get(block=False)
                    except queue.Empty:
                        # No item available, move to next worker immediately
                        current_worker = (current_worker + 1) % self.workers_count
                        continue
                else:
                    # Blocking: wait for item (strict round-robin)
                    item = self._results_queues[current_worker].get(block=True, timeout=5.0)
                # Put the item into the shared results queue
                if not isinstance(item, VentilatedItemProcessedMessage):
                    # Skip VentilatedItemProcessedMessage - it's just a control message
                    self._shared_results_queue.put(item, block=False) 
                
                # Move to next worker in round-robin fashion
                current_worker = (current_worker + 1) % self.workers_count
            except queue.Empty:
                # No item available from current worker, move to next
                current_worker = (current_worker + 1) % self.workers_count
                continue
            except queue.Full:
                # Shared queue is full, wait a bit and try again
                continue
            except Exception as e:
                # Any other exception, continue to next worker
                current_worker = (current_worker + 1) % self.workers_count
                continue
        
    def results_qsize(self):
        return self._shared_results_queue.qsize()

    @property
    def diagnostics(self):
        return {'output_queue_size': self.results_qsize()}