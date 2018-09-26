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

from time import sleep

from petastorm.reader import ReaderV2


def test_bound_size_of_output_queue_size_reader_v2(synthetic_dataset):
    """Limit the output queue size to 5. Make sure the reader does not overrun the target output queue size.
    This test is timing sensitive so it might become flaky"""
    TIME_TO_GET_TO_OUTPUT_QUEUE_OF_5 = 3.0
    DELTA_T = 0.1

    with ReaderV2(synthetic_dataset.url, target_output_queue_size=5) as reader:
        next(reader)
        for _ in range(int(round(TIME_TO_GET_TO_OUTPUT_QUEUE_OF_5 / DELTA_T))):
            if reader.diagnostics['output_queue_size'] == 5:
                sleep(1.0)
                assert reader.diagnostics['output_queue_size'] == 5
                return
            sleep(DELTA_T)
    assert False, 'Was not able to get to 5 items in the output queue withing '
