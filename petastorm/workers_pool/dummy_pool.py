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

from time import sleep

from petastorm.workers_pool import EmptyResultError


class DummyPool(object):
    """This class has pool interface but performs all work in calls to get_results. It is sometimes convenient
    to substitute a real pool with this dummy implementation.

    Found this class useful when profiling worker code. When on a separate thread, the worker code was not observable
    (out of the box) by the profiler"""

    # Have workers argument just to make compatible with other pool implementations
    def __init__(self, workers=None):
        # We just accumulate all ventilated items in the list
        self._ventilator_queue = []

        # get_results will populate this list
        self._results_queue = []
        self._worker = None
        self._ventilator = None
        self.workers_count = 1

    def start(self, worker_class, worker_args=None, ventilator=None):
        # Instantiate a single worker with all the args
        self._worker = worker_class(0, self._results_queue.append, worker_args)

        if ventilator:
            self._ventilator = ventilator
            self._ventilator.start()

    def ventilate(self, *args, **kargs):
        """Send a work item to a worker process."""
        self._ventilator_queue.append((args, kargs))

    def get_results(self):
        """Returns results

        The processing is done on the get_results caller thread if the results queue is empty

        :return: arguments passed to publish_func(...) by a worker
        """

        if self._results_queue:
            # We have already calculated result. Just return it
            return self._results_queue.pop(0)
        else:
            # If we don't have any tasks waiting for processing, then indicate empty queue
            while self._ventilator_queue or (self._ventilator and not self._ventilator.completed()):

                # To prevent a race condition of the ventilator working but not yet placing an item
                # on the ventilator queue. We block until something is on the ventilator queue.
                while not self._ventilator_queue:
                    sleep(.1)

                # If we do have some tasks, then process a task from the head of a queue
                args, kargs = self._ventilator_queue.pop(0)
                self._worker.process(*args, **kargs)

                if self._ventilator:
                    self._ventilator.processed_item()

                if self._results_queue:
                    return self._results_queue.pop(0)

            raise EmptyResultError()

    def stop(self):
        if self._ventilator:
            self._ventilator.stop()

    def join(self):
        pass

    @property
    def diagnostics(self):
        return dict()
