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

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import ProcessPoolExecutor

from petastorm.reader_impl.pool_helpers import submit_with_dill
from petastorm.reader_impl.same_thread_executor import SameThreadExecutor


def test_thread_pool_submit():
    for executor in [ProcessPoolExecutor(2), SameThreadExecutor(), ThreadPoolExecutor(2)]:
        future = submit_with_dill(executor, lambda x: x * 2, 10)
        assert future.result() == 20
