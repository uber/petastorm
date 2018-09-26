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
from __future__ import division

import concurrent.futures
import logging
import sys
import traceback
from collections import Counter
from time import sleep, time
from traceback import format_exception

from six.moves import queue

logger = logging.getLogger(__name__)

CHUNK_DECODING_SIZE = 10

_MAX_IN_LOADING_QUEUE_SIZE = 14
_MAX_IN_DECODING_QUEUE_SIZE = 14
_LOOP_IDLE_TIME_SEC = 0.001  # 10 ms

_OUTPUT_QUEUE_SIZE = 30


class RateCalculator(object):
    def __init__(self, name):
        self._name = name
        self.rate = None
        self._last_measured_time = None
        self._last_count = 0

    def update(self, count):
        now = time()
        if not self._last_measured_time:
            self._last_measured_time = now
            self._last_count = count

        elif self._last_measured_time + 2.0 < now:
            interval = now - self._last_measured_time
            self.rate = (count - self._last_count) / interval
            self._last_measured_time = now
            self._last_count = count

    def __str__(self):
        return '{}: {} fps'.format(self._name, self.rate)


class EOFSentinel(object):
    """This is a 'poison' object being queued into the output queue to signal that no further data will be arriving.
    This would happen when the number of epochs is limited and we are done reading all the data."""
    pass


def _decode_row_dispatcher(decoder, row, row_serializer):
    decoded_row = decoder.decode(row)
    if row_serializer:
        return row_serializer.serialize(decoded_row)
    else:
        return decoded_row


class WorkerLoopError(Exception):
    def __init__(self, inner_error, exc_info):
        self.inner_error = inner_error
        self.exc_info = exc_info

    def __str__(self):
        return 'worker_loop has failed with {}. Call stack:\n{}'.format(self.inner_error,
                                                                        ''.join(format_exception(*self.exc_info)))


# Runs in a separate thread
def dispatch_decode(decoder, shuffled):
    return decoder.decode(shuffled)


def worker_loop(epochs_generator, loading_pool, loader, decoding_pool, decoder, shuffling_queue, output_queue,
                stop_event, row_serializer, stats):
    # Possibly infinite loop driven by row-group ventilation logic (infinite if epochs=inf)
    in_loading_futures = []
    in_decoding = []

    if not isinstance(stats, Counter):
        raise ValueError('stats argument is expected to be a collections.Counter')

    chunk_decoding_size = max(1, decoding_pool._max_workers / _OUTPUT_QUEUE_SIZE)

    try:
        epochs_iterator = iter(epochs_generator())
        # rowgroup_spec is currently a ParquetDatasetPiece, but will likely be augmented with petastorm specific info
        # in the future
        try:
            rowgroup_spec = next(epochs_iterator)
        except StopIteration:
            rowgroup_spec = None

        load_rate = RateCalculator('load')
        from_shuffling = RateCalculator('from_shuffling')
        from_decoding = RateCalculator('from_decoding')

        # Continue iterating until we have some data in one of the queues connecting loader/shuffling-queue/decoder or
        # we still have some rowgroup_specs to process
        while rowgroup_spec is not None or in_loading_futures or in_decoding or shuffling_queue.size > 0:
            # An owner can signal a stop before we are actually finished
            if stop_event.is_set():
                break

            # 1. Read more rowgroup_specs from our epochs generator and schedule 'loader' to load these rowgroups
            # 2. Readout what was already loaded and enqueue loaded rows into a shuffling queue
            # 3. Read from the shuffling queue and submit decoding tasks to decoding executor
            # 4. Read completed decoding futures and put the results into the final output queue

            # 1. Read more rowgroup_specs from our epochs generator and schedule 'loader' to load these rowgroups
            # ---------------------------------------------------------------------------------------------------

            # Need to keep a cap (_MAX_IN_LOADING_QUEUE_SIZE) on the queue so we don't get unbounded growth with
            # infinite epochs.
            if rowgroup_spec is not None and len(in_loading_futures) < _MAX_IN_LOADING_QUEUE_SIZE:
                in_loading_futures.append(loading_pool.submit(loader.load, rowgroup_spec))
                stats['0_rowgroups_scheduled_for_loading'] += 1

                try:
                    rowgroup_spec = next(epochs_iterator)
                except StopIteration:
                    rowgroup_spec = None

            # 2. Readout what was already loaded and enqueue loaded rows into a shuffling queue
            # ---------------------------------------------------------------------------------

            # Futures that have completed will be placed first in the returned list
            loaded_futures = concurrent.futures.as_completed(in_loading_futures, timeout=0)

            # Push all loaded_futures and partially decoded data into shuffling queue
            while shuffling_queue.can_add():
                try:
                    loaded_future = next(loaded_futures)
                    loaded_future_result = loaded_future.result()
                    stats['rows_loaded'] += len(loaded_future_result)
                    load_rate.update(stats['rows_loaded'])
                    shuffling_queue.add_many(loaded_future_result)

                    # Done processing, remove from the queue. 'remove' on the list is ok since we cap the list
                    # to a small size with _MAX_IN_LOADING_QUEUE_SIZE
                    in_loading_futures.remove(loaded_future)
                except (concurrent.futures.TimeoutError, StopIteration):
                    break

            # 3. Read from the shuffling queue and submit decoding tasks to decoding executor
            # ---------------------------------------------------------------------------------
            while shuffling_queue.can_retrieve() and len(in_decoding) < _MAX_IN_DECODING_QUEUE_SIZE:
                shuffled = shuffling_queue.retrieve_many(chunk_decoding_size)
                stats['rows_read_from_shuffling_queue'] += len(shuffled)
                from_shuffling.update(stats['rows_read_from_shuffling_queue'])
                in_decoding.append(decoding_pool.submit(_decode_row_dispatcher, decoder, shuffled, row_serializer))

            # 4. Read completed decoding futures and put the results into the final output queue
            # ---------------------------------------------------------------------------------

            # We want to keep the order in the output queue the same as in the decoding input. This determinism
            # is useful for testing purposes and if we actually want to read in order (need to use
            # SingleThreadExecutor for that though)
            decoded = concurrent.futures.as_completed(in_decoding, timeout=0)

            # We'll keep a list of finished futures (already_decoded) in a set, but would follow the order of futures
            # in the `in_decoding` list when writing to output_queue

            already_decoded = set()
            while True:
                try:
                    next_decoded = next(decoded)
                    stats['row_batches_read_from_decoder'] += 1
                    already_decoded.add(next_decoded)
                except (concurrent.futures.TimeoutError, StopIteration):
                    break

            stats['in_decoding_size'] = len(in_decoding)

            updated_in_decoding = []
            for i, was_decoding in enumerate(in_decoding):
                if was_decoding in already_decoded:
                    stats['rows_written_to_output_queue'] += 1
                    while not stop_event.is_set():
                        try:
                            buffer = was_decoding.result()
                            was_decoding_result = row_serializer.deserialize(buffer) if row_serializer else buffer
                            output_queue.put(was_decoding_result, block=False)
                            stats['rows_written_to_output_queue'] += len(was_decoding_result)
                            from_decoding.update(stats['rows_written_to_output_queue'])
                            break
                        except queue.Full:
                            raise RuntimeError('Do not set Queue\'s maxsize. The size of the queue is managed by '
                                               'worker_loop by design')
                else:
                    updated_in_decoding.append(in_decoding[i])

            in_decoding = updated_in_decoding
            stats['output_queue_size'] = output_queue.qsize()
            stats['in_decoding_size'] = len(in_decoding)
            sleep(_LOOP_IDLE_TIME_SEC)

        # If we were ordered to stop, better not write the sentinel out since the queue can be full
        # and we would block forever. There should not be an expectation from the caller to get a sentinel in this case
        if not stop_event.is_set():
            output_queue.put(EOFSentinel())

    except Exception as e:  # pylint: disable=broad-except
        logger.error(traceback.format_exc())
        output_queue.put(WorkerLoopError(e, sys.exc_info()))
