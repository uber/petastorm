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
import concurrent.futures
import logging
import sys
import traceback
from collections import Counter
from time import sleep
from traceback import format_exception

from six.moves import queue

logger = logging.getLogger(__name__)

_MAX_IN_LOADING_QUEUE_SIZE = 14
_MAX_IN_DECODING_QUEUE_SIZE = 14
_LOOP_IDLE_TIME_SEC = 0.001  # 1 ms


class EOFSentinel(object):
    """This is a 'poison' object being queued into the output queue to signal that no further data will be arriving.
    This would happen when the number of epochs is limited and we are done reading all the data."""


def _decode_row_dispatcher(decoder, row):
    return decoder.decode(row)


class WorkerLoopError(Exception):
    def __init__(self, inner_error, exc_info):
        self.inner_error = inner_error
        self.exc_info = exc_info

    def __str__(self):
        return 'worker_loop has failed with {}. Call stack:\n{}'.format(self.inner_error,
                                                                        ''.join(format_exception(*self.exc_info)))


# Runs in a separate thread
def worker_loop(epochs_generator, loading_pool, loader, decoding_pool, decoder, shuffling_queue, output_queue,
                stop_event, stats):
    # Possibly infinite loop driven by row-group ventilation logic (infinite if epochs=inf)
    in_loading_futures = []
    in_decoding = []

    if not isinstance(stats, Counter):
        raise ValueError('stats argument is expected to be a collections.Counter')

    try:
        epochs_iterator = iter(epochs_generator())
        # rowgroup_spec is currently a ParquetDatasetPiece, but will likely be augmented with petastorm specific info
        # in the future
        try:
            rowgroup_spec = next(epochs_iterator)
        except StopIteration:
            rowgroup_spec = None

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
                stats['rowgroups_scheduled_for_loading'] += 1
                stats['in_loading_queue_len'] = len(in_loading_futures)

                try:
                    rowgroup_spec = next(epochs_iterator)
                except StopIteration:
                    rowgroup_spec = None

            # 2. Readout what was already loaded and enqueue loaded rows into a shuffling queue
            # ---------------------------------------------------------------------------------

            # Futures that have completed will be placed first in the returned list
            loaded_futures = concurrent.futures.as_completed(in_loading_futures, timeout=0)

            # If no more data is ever expected from the loading, we should let shuffling buffer
            # deplete itself. Otherwise, decoding will wait forever for data from shuffling that
            # will never come.
            if not in_loading_futures and not rowgroup_spec:
                shuffling_queue.finish()

            # Push all loaded_futures and partially decoded data into shuffling queue
            while shuffling_queue.can_add():
                try:
                    loaded_future = next(loaded_futures)
                    loaded_future_result = loaded_future.result()
                    stats['rows_loaded'] += len(loaded_future_result)
                    stats['shuffling_buffer_size'] = shuffling_queue.size
                    shuffling_queue.add_many(loaded_future_result)

                    # Done processing, remove from the queue. 'remove' on the list is ok since we cap the list
                    # to a small size with _MAX_IN_LOADING_QUEUE_SIZE
                    in_loading_futures.remove(loaded_future)
                except (concurrent.futures.TimeoutError, StopIteration):
                    break

            # 3. Read from the shuffling queue and submit decoding tasks to decoding executor
            # ---------------------------------------------------------------------------------
            while shuffling_queue.can_retrieve() and len(in_decoding) < _MAX_IN_DECODING_QUEUE_SIZE:
                shuffled = shuffling_queue.retrieve()
                stats['rows_read_from_shuffling_queue'] += 1
                in_decoding.append(decoding_pool.submit(_decode_row_dispatcher, decoder, shuffled))
                stats['in_decoding_queue_len'] = len(in_decoding)

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
                    stats['rows_read_from_decoder'] += 1
                    already_decoded.add(next_decoded)
                except (concurrent.futures.TimeoutError, StopIteration):
                    break

            # follow the order of futures in the `in_decoding` list
            updated_in_decoding = []
            for i, was_decoding in enumerate(in_decoding):
                if was_decoding in already_decoded:
                    stats['rows_written_to_output_queue'] += 1
                    while not stop_event.is_set():
                        try:
                            output_queue.put(was_decoding.result(), block=False)
                            break
                        except queue.Full:
                            sleep(_LOOP_IDLE_TIME_SEC)
                else:
                    updated_in_decoding.append(in_decoding[i])

            in_decoding = updated_in_decoding
            stats['output_queue_size'] = output_queue.qsize()
            sleep(_LOOP_IDLE_TIME_SEC)

        # If we were ordered to stop, better not write the sentinel out since the queue can be full
        # and we would block forever. There should not be an expectation from the caller to get a sentinel in this case
        if not stop_event.is_set():
            output_queue.put(EOFSentinel())

    except Exception as e:  # pylint: disable=broad-except
        logger.error(traceback.format_exc())
        output_queue.put(WorkerLoopError(e, sys.exc_info()))
