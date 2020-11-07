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

"""This command line utility instantiates an instance of a Reader and measures its throughput. """

from __future__ import division
from __future__ import print_function

import argparse
import logging
import sys

from petastorm.benchmark.throughput import reader_throughput, \
    WorkerPoolType, ReadMethod

logger = logging.getLogger(__name__)


def _parse_args(args):
    # If min-after-dequeue value is not explicitly set from the command line, it will be calculated from the total
    # shuffling queue size multiplied by this ratio
    DEFAULT_MIN_AFTER_DEQUEUE_TO_QUEUE_SIZE_RATIO = 0.8

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('dataset_path', type=str, help='Path to a petastorm dataset')
    parser.add_argument('--field-regex', type=str, nargs='+',
                        help='A list of regular expressions. Only fields that match one of the regex patterns will '
                             'be used during the benchmark.')

    parser.add_argument('-w', '--workers-count', type=int, default=3,
                        help='Number of workers used by the reader')
    parser.add_argument('-p', '--pool-type', type=WorkerPoolType, default=WorkerPoolType.THREAD,
                        choices=list(WorkerPoolType),
                        help='Type of a worker pool used by the reader')

    parser.add_argument('-m', '--warmup-cycles', type=int, default=200,
                        help='Number of warmup read cycles. Warmup read cycles run before measurement cycles and '
                             'the throughput during these cycles is not accounted for in the reported results.')
    parser.add_argument('-n', '--measure-cycles', type=int, default=1000,
                        help='Number cycles used for benchmark measurements. Measurements cycles are run after '
                             'warmup cycles.')

    parser.add_argument('--profile-threads', dest='profile_threads', action='store_true',
                        help='Enables profiling threads. Will print result when thread pool is shut down.')

    parser.add_argument('-d', '--read-method', type=ReadMethod, choices=list(ReadMethod),
                        default=ReadMethod.PYTHON,
                        help='Which read mode to use: \'python\': using python implementation. '
                             '\'tf\': constructing a small TF graph streaming data from pure python implementation.')

    parser.add_argument('-q', '--shuffling-queue-size', type=int, default=500, required=False,
                        help='Size of the shuffling queue used to decorrelate row-group chunks. ')

    parser.add_argument('--min-after-dequeue', type=int, default=None, required=False,
                        help='Minimum number of elements in a shuffling queue before entries can be read from it. '
                             'Default value is set to {}%% of the --shuffling-queue-size '
                             'parameter'.format(100 * DEFAULT_MIN_AFTER_DEQUEUE_TO_QUEUE_SIZE_RATIO))

    parser.add_argument('-vv', action='store_true', default=False, help='Sets logging level to DEBUG.')
    parser.add_argument('-v', action='store_true', default=False, help='Sets logging level to INFO.')

    args = parser.parse_args(args)

    if not args.min_after_dequeue:
        args.min_after_dequeue = DEFAULT_MIN_AFTER_DEQUEUE_TO_QUEUE_SIZE_RATIO * args.shuffling_queue_size

    return args


def _main(args):
    logging.basicConfig()
    args = _parse_args(args)

    if args.v:
        logging.getLogger().setLevel(logging.INFO)
    if args.vv:
        logging.getLogger().setLevel(logging.DEBUG)

    results = reader_throughput(args.dataset_path, args.field_regex, warmup_cycles_count=args.warmup_cycles,
                                measure_cycles_count=args.measure_cycles, pool_type=args.pool_type,
                                loaders_count=args.workers_count, profile_threads=args.profile_threads,
                                read_method=args.read_method, shuffling_queue_size=args.shuffling_queue_size,
                                min_after_dequeue=args.min_after_dequeue)

    logger.info('Done')
    print('Average sample read rate: {:1.2f} samples/sec; RAM {:1.2f} MB (rss); '
          'CPU {:1.2f}%'.format(results.samples_per_second, results.memory_info.rss / 2 ** 20, results.cpu))


def main():
    _main(sys.argv[1:])


if __name__ == '__main__':
    _main(sys.argv[1:])
