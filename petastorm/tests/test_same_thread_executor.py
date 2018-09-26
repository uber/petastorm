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

from concurrent.futures import as_completed

from petastorm.reader_impl.same_thread_executor import SameThreadExecutor


def test_single_future_submit():
    e = SameThreadExecutor()
    assert 10 == e.submit(lambda: 10).result()


def test_many_future_submit():
    e = SameThreadExecutor()
    futures = []
    for x in range(10):
        futures.append(e.submit(lambda y=x: y))

    completed = as_completed(futures)
    results = [f.result() for f in completed]
    assert list(range(10)) == sorted(results)


def test_max_workers():
    assert SameThreadExecutor()._max_workers == 1
