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

import copy
import logging
import re
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum

import psutil
import six
import tensorflow as tf

from petastorm.etl.dataset_metadata import get_schema_from_dataset_url
from petastorm.reader import Reader, ReaderV2
from petastorm.reader_impl.same_thread_executor import SameThreadExecutor
from petastorm.reader_impl.shuffling_buffer import RandomShufflingBuffer
from petastorm.tf_utils import tf_tensors
from petastorm.workers_pool.dummy_pool import DummyPool
from petastorm.workers_pool.process_pool import ProcessPool
from petastorm.workers_pool.thread_pool import ThreadPool

logger = logging.getLogger(__name__)

BenchmarkResult = namedtuple('BenchmarkResult', ['time_mean', 'samples_per_second', 'memory_info', 'cpu'])


class WorkerPoolType(Enum):
    """Defines a type of parallelism used in the benchmark: multithreading, multiprocessing or none (single-thread)"""
    THREAD = 'thread'
    """A thread pool is used by the benchmark"""

    PROCESS = 'process'
    """A process pool is used by the benchmark"""

    NONE = 'none'
    """IO and loading will be done on a single thread. No parallelism."""

    def __str__(self):
        return self.value


class ReadMethod(Enum):
    """Defines whether a Tensorflow or plain Python reading method would be used during the benchmark"""
    TF = 'tf'
    """Tensorflow reading method will be used during the benchmark (``tf_tensor`` method)"""

    PYTHON = 'python'
    """Pure python reading method will be used during the benchmark (``next(reader)``)"""

    def __str__(self):
        return self.value


def _time_warmup_and_work(reader, warmup_cycles_count, measure_cycles_count, do_work_func=None):

    if not do_work_func:
        do_work_func = lambda: next(reader)  # noqa

    _time_multiple_iterations(warmup_cycles_count, do_work_func, lambda: reader.diagnostics)

    logger.info('Done warmup')

    this_process = psutil.Process()
    this_process.cpu_percent()

    duration = _time_multiple_iterations(measure_cycles_count, do_work_func, lambda: reader.diagnostics)

    cpu_percent = this_process.cpu_percent()

    time_mean = duration / measure_cycles_count
    result = BenchmarkResult(time_mean=time_mean,
                             samples_per_second=1.0 / time_mean,
                             memory_info=this_process.memory_full_info(),
                             cpu=cpu_percent)
    logger.info('Done measuring: %s', str(result))

    return result


def _time_warmup_and_work_tf(reader, warmup_cycles_count, measure_cycles_count, shuffling_queue_size,
                             min_after_dequeue):
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        readout_tensors = tf_tensors(reader, shuffling_queue_size, min_after_dequeue)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, start=True, sess=sess)

        result = _time_warmup_and_work(reader, warmup_cycles_count, measure_cycles_count,
                                       lambda: sess.run(readout_tensors))

        coord.request_stop()
        coord.join(threads)

    return result


def reader_throughput(dataset_url, field_regex=None, warmup_cycles_count=300, measure_cycles_count=1000,
                      pool_type=WorkerPoolType.THREAD, loaders_count=3, profile_threads=False,
                      read_method=ReadMethod.PYTHON, shuffling_queue_size=500, min_after_dequeue=400,
                      reader_extra_args=None, spawn_new_process=True):
    """Constructs a Reader instance and uses it to performs throughput measurements.

    The function will spawn a new process if ``spawn_separate_process`` is set. This is needed to make memory footprint
    measurements accurate.

    :param dataset_url: A url of the dataset to be used for measurements.
    :param field_regex:  A list of regular expressions. Only fields that match one of the regex patterns will be used
      during the benchmark.
    :param warmup_cycles_count: Number of warmup cycles. During warmup cycles no measurements are being recorded.
    :param measure_cycles_count: Number of measurements cycles. Only time elapsed during measurements cycles are used
      in throughput calculations.
    :param pool_type: :class:`WorkerPoolType` enum value.
    :param loaders_count: Number of threads (same thread is used for IO and decoding).
    :param profile_threads:  Enables profiling threads. Will print result when thread pool is shut down.
    :param read_method:  An enum :class:`ReadMethod` that defines whether a :class:`petastorm.reader.Reader` will be
      used.
    :param shuffling_queue_size: Maximum number of elements in the shuffling queue.
    :param min_after_dequeue: Minimum number of elements in a shuffling queue before entries can be read from it.
    :param reader_extra_args: Extra arguments that would be passed to Reader constructor.
    :param spawn_new_process: This function will respawn itself in a new process if the argument is True. Spawning
      a new process is needed to get an accurate memory footprint.

    :return: An instance of ``BenchmarkResult`` namedtuple with the results of the benchmark. The namedtuple has
      the following fields: `time_mean`, `samples_per_second`, `memory_info` and `cpu`
    """
    if not reader_extra_args:
        reader_extra_args = dict()

    if spawn_new_process:
        args = copy.deepcopy(locals())
        args['spawn_new_process'] = False
        executor = ProcessPoolExecutor(1)
        future = executor.submit(reader_throughput, **args)
        return future.result()

    logger.info('Arguments: %s', locals())

    if 'schema_fields' not in reader_extra_args:
        unischema_fields = _filter_schema_fields(get_schema_from_dataset_url(dataset_url), field_regex)
        reader_extra_args['schema_fields'] = unischema_fields

    logger.info('Fields used in the benchmark: %s', str(reader_extra_args['schema_fields']))

    with Reader(dataset_url,
                num_epochs=None,
                reader_pool=_create_worker_pool(pool_type, loaders_count, profile_threads),
                **reader_extra_args) as reader:

        if read_method == ReadMethod.PYTHON:
            result = _time_warmup_and_work(reader, warmup_cycles_count, measure_cycles_count)
        elif read_method == ReadMethod.TF:
            result = _time_warmup_and_work_tf(reader, warmup_cycles_count, measure_cycles_count,
                                              shuffling_queue_size, min_after_dequeue)
        else:
            raise RuntimeError('Unexpected reader_type value: %s', str(read_method))

    return result


def reader_v2_throughput(dataset_url, field_regex=None, warmup_cycles_count=300, measure_cycles_count=1000,
                         pool_type=WorkerPoolType.THREAD, loaders_count=3, decoders_count=3,
                         read_method=ReadMethod.PYTHON, shuffling_queue_size=500, min_after_dequeue=400,
                         reader_extra_args=None, spawn_new_process=True):
    """Constructs a ReaderV2 instance and uses it to performs throughput measurements.

    The function will spawn a new process if ``spawn_separate_process`` is set. This is needed to make memory footprint
    measurements accurate.

    :param dataset_url: A url of the dataset to be used for measurements.
    :param field_regex:  A list of regular expressions. Only fields that match one of the regex patterns will be used
      during the benchmark.
    :param warmup_cycles_count: Number of warmup cycles. During warmup cycles no measurements are being recorded.
    :param measure_cycles_count: Number of measurements cycles. Only time elapsed during measurements cycles are used
      in throughput calculations.
    :param pool_type: :class:`WorkerPoolType` enum value.
    :param loaders_count: Number of IO threads.
    :param decoders_count: Number of threads or processes used for decoding. ``pool_type`` parameter defines
      whether multiple processes or threads are used for parallel decoding.
    :param read_method:  An enum :class:`ReadMethod` that defines whether a :class:`petastorm.reader.Reader` will be
      used.
    :param shuffling_queue_size: Maximum number of elements in the shuffling queue.
    :param min_after_dequeue: Minimum number of elements in a shuffling queue before entries can be read from it.
    :param reader_extra_args: Extra arguments that would be passed to Reader constructor.
    :param spawn_new_process: This function will respawn itself in a new process if the argument is True. Spawning
      a new process is needed to get an accurate memory footprint.

    :return: An instance of ``BenchmarkResult`` namedtuple with the results of the benchmark. The namedtuple has
      the following fields: `time_mean`, `samples_per_second`, `memory_info` and `cpu`
    """
    if not reader_extra_args:
        reader_extra_args = dict()

    if spawn_new_process:
        args = copy.deepcopy(locals())
        args['spawn_new_process'] = False
        executor = ProcessPoolExecutor(1)
        future = executor.submit(reader_v2_throughput, **args)
        return future.result()

    logger.info('Arguments: %s', locals())

    if 'schema_fields' not in reader_extra_args:
        unischema_fields = _filter_schema_fields(get_schema_from_dataset_url(dataset_url), field_regex)
        reader_extra_args['schema_fields'] = unischema_fields

    logger.info('Fields used in the benchmark: %s', str(reader_extra_args['schema_fields']))

    decoder_pool_executor = _create_concurrent_executor(pool_type, decoders_count)

    with ReaderV2(dataset_url, num_epochs=None,
                  loader_pool=ThreadPoolExecutor(loaders_count),
                  decoder_pool=decoder_pool_executor,
                  shuffling_queue=RandomShufflingBuffer(shuffling_queue_size, min_after_dequeue),
                  **reader_extra_args) as reader:

        if read_method == ReadMethod.PYTHON:
            result = _time_warmup_and_work(reader, warmup_cycles_count, measure_cycles_count)
        elif read_method == ReadMethod.TF:
            result = _time_warmup_and_work_tf(reader, warmup_cycles_count, measure_cycles_count, 0, 0)
        else:
            raise RuntimeError('Unexpected reader_type value: %s', str(read_method))

    return result


def _create_concurrent_executor(pool_type, decoders_count):
    if pool_type == WorkerPoolType.PROCESS:
        decoder_pool_executor = ProcessPoolExecutor(decoders_count)
    elif pool_type == WorkerPoolType.THREAD:
        decoder_pool_executor = ThreadPoolExecutor(decoders_count)
    elif pool_type == WorkerPoolType.NONE:
        decoder_pool_executor = SameThreadExecutor()
    else:
        raise ValueError('Unexpected pool type value: %s', pool_type)
    return decoder_pool_executor


def _create_worker_pool(pool_type, workers_count, profiling_enabled):
    """Different worker pool implementation (in process none or thread-pool, out of process pool)"""
    if pool_type == WorkerPoolType.THREAD:
        worker_pool = ThreadPool(workers_count, profiling_enabled=profiling_enabled)
    elif pool_type == WorkerPoolType.PROCESS:
        worker_pool = ProcessPool(workers_count)
    elif pool_type == WorkerPoolType.NONE:
        worker_pool = DummyPool()
    else:
        raise ValueError('Supported pool types are thread, process or dummy. Got {}.'.format(pool_type))
    return worker_pool


def _filter_schema_fields(schema, field_regex):
    if field_regex:
        unischema_fields = []
        for pattern in field_regex:
            unischema_fields.extend(
                [field for field_name, field in schema.fields.items() if re.match(pattern, field_name)])
    else:
        unischema_fields = field_regex
    return unischema_fields


def _time_multiple_iterations(iterations, work_func, diags_info_func=None, report_period=1.0):
    start_time = time.time()
    last_reported_time = start_time
    last_reported_count = 0

    for current_cycle in six.moves.xrange(iterations):
        work_func()
        now = time.time()
        eps = 1e-9
        if now - last_reported_time > report_period:
            message = '{:2.2f} (mean: {:2.2f}) iterations/sec.' \
                .format(float(current_cycle - last_reported_count) / (eps + now - last_reported_time),
                        float(current_cycle) / (eps + now - start_time))
            last_reported_count = current_cycle
            last_reported_time = now
            if diags_info_func:
                message += ' diags:{}'.format(str(diags_info_func()))
            logging.debug(message)

    return time.time() - start_time
