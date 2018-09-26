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

logger = logging.getLogger(__name__)

CHUNK_DECODING_SIZE = 10

_MAX_IN_LOADING_QUEUE_SIZE = 14
_MAX_IN_DECODING_QUEUE_SIZE = 14
_LOOP_IDLE_TIME_SEC = 0.01  # 10 ms


class RateCalculator(object):
    def __init__(self, name):
        self._name = name
        self.rate = -1
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


def _decode_rows_dispatcher(decoder, rows, row_serializer):
    decoded_rows = decoder.decode(rows)
    if row_serializer:
        return row_serializer.serialize(decoded_rows)
    else:
        return decoded_rows


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


def _x(blocking):
    return 'X' if blocking else '-'


def _render_stats(stats):
    now = time()

    last_report = _render_stats.last_report if hasattr(_render_stats, 'last_report') else 0
    # print(hasattr(_render_stats, 'last_report'), last_report)

    if last_report + 1.0 < now:
        stats_string = \
            ('E{} ' +
             'L:{}|{} ' +
             'S:{}|{}|{}|{} ' +
             'D:{}|{}|{}|{} ' +
             'O:{}|{}|{}|{}').format(_x(stats['epochs_ended']),
                                     _x(stats['loader_full']), stats['loader_rate'],
                                     _x(stats['shuffling_can_add']), stats['shuffling_size'],
                                     stats['shuffling_rows_retrieved_rate'], _x(stats['shuffling_can_retrieve']),
                                     _x(stats['decoding_full']), stats['chunk_decoding_size'],
                                     stats['decoding_in_process'], stats['decoder_batches_read'],
                                     _x(stats['output_full']), stats['output_queue_size'],
                                     stats['output_rows_put_rate'], stats['output_rows_put'])

        logger.debug(stats_string)

        _render_stats.last_report = now
    # print('2', hasattr(_render_stats, 'last_report'), last_report)


def worker_loop(epochs_generator, schema, loading_pool, loader, decoding_pool, decoder, shuffling_queue, output_queue,
                stop_event, row_serializer, target_output_queue_size, stats):
    # Possibly infinite loop driven by row-group ventilation logic (infinite if epochs=inf)
    in_loading_futures = []
    in_decoding = []

    if not isinstance(stats, Counter):
        raise ValueError('stats argument is expected to be a collections.Counter')

    # This is the size of the group that would be sent to the decoder. We update it later, when we know the
    # row-group size
    chunk_decoding_size = 50

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
                stats['loader_total_rowgroup_scheduled'] += 1

                try:
                    rowgroup_spec = next(epochs_iterator)
                    stats['epochs_ended'] = False
                except StopIteration:
                    stats['epochs_ended'] = True
                    rowgroup_spec = None
            stats['loader_full'] = len(in_loading_futures) >= _MAX_IN_LOADING_QUEUE_SIZE

            # 2. Readout what was already loaded and enqueue loaded rows into a shuffling queue
            # ---------------------------------------------------------------------------------

            # Futures that have completed will be placed first in the returned list
            loaded_futures = concurrent.futures.as_completed(in_loading_futures, timeout=0)

            # Push all loaded_futures and partially decoded data into shuffling queue
            while shuffling_queue.can_add():
                try:
                    loaded_future = next(loaded_futures)
                    loaded_future_result = loaded_future.result()

                    # Not necessary an optimal choice of chunk_decoding_size, but should handle the cases of
                    # large rowgroups with many small records
                    stats['chunk_decoding_size'] = chunk_decoding_size

                    stats['loader_rows_loaded'] += len(loaded_future_result)
                    load_rate.update(stats['loader_rows_loaded'])
                    shuffling_queue.add_many(loaded_future_result)

                    # Done processing, remove from the queue. 'remove' on the list is ok since we cap the list
                    # to a small size with _MAX_IN_LOADING_QUEUE_SIZE
                    in_loading_futures.remove(loaded_future)
                except (concurrent.futures.TimeoutError, StopIteration):
                    break

            stats['shuffling_can_add'] = shuffling_queue.can_add()

            # 3. Read from the shuffling queue and submit decoding tasks to decoding executor
            # ---------------------------------------------------------------------------------
            while shuffling_queue.can_retrieve() and len(in_decoding) < _MAX_IN_DECODING_QUEUE_SIZE:
                shuffled = shuffling_queue.retrieve_many(chunk_decoding_size)
                stats['shuffling_rows_retrieved'] += len(shuffled)
                stats['shuffling_size'] = shuffling_queue.size
                from_shuffling.update(stats['shuffling_rows_retrieved'])
                in_decoding.append(decoding_pool.submit(_decode_rows_dispatcher, decoder, shuffled, row_serializer))

            stats['shuffling_can_retrieve'] = shuffling_queue.can_retrieve()
            stats['decoding_full'] = len(in_decoding) >= _MAX_IN_DECODING_QUEUE_SIZE

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
                    stats['decoder_batches_read'] += 1
                    already_decoded.add(next_decoded)
                except (concurrent.futures.TimeoutError, StopIteration):
                    break

            stats['decoding_batches_in_process'] = len(in_decoding)
            stats['decoding_results_ready'] = len(already_decoded) > 0

            updated_in_decoding = []
            if len(already_decoded) > 0:
                for i, was_decoding in enumerate(in_decoding):
                    if output_queue.qsize() * chunk_decoding_size < target_output_queue_size:
                        stats['output_full'] = False
                        if was_decoding in already_decoded:
                            result_buffer = was_decoding.result()
                            if row_serializer:
                                was_decoding_result = row_serializer.deserialize(result_buffer)
                            else:
                                was_decoding_result = result_buffer
                            stats['output_batches_put'] += 1
                            stats['output_rows_put'] += len(was_decoding_result)
                            output_queue.put(was_decoding_result, block=False)
                            from_decoding.update(stats['output_rows_put'])
                        else:
                            updated_in_decoding.append(in_decoding[i])
                    else:
                        stats['output_full'] = True
                        updated_in_decoding.append(in_decoding[i])

                in_decoding = updated_in_decoding
                stats['output_queue_size'] = output_queue.qsize()
                stats['decoding_in_process'] = len(in_decoding)
            stats['loader_rate'] = load_rate.rate
            stats['shuffling_rows_retrieved_rate'] = from_shuffling.rate
            stats['output_rows_put_rate'] = from_decoding.rate
            _render_stats(stats)
            sleep(_LOOP_IDLE_TIME_SEC)

        # If we were ordered to stop, better not write the sentinel out since the queue can be full
        # and we would block forever. There should not be an expectation from the caller to get a sentinel in this case
        if not stop_event.is_set():
            output_queue.put(EOFSentinel())

    except Exception as e:  # pylint: disable=broad-except
        logger.error(traceback.format_exc())
        output_queue.put(WorkerLoopError(e, sys.exc_info()))
