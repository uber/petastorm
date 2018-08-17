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

"""A mock executor. Guarantees stable order of future evaluations: a submitted future is
evaluated immediately as part of submit implementation."""

from concurrent.futures import Executor
from concurrent.futures._base import FINISHED


class _FakeCondition(object):
    def acquire(self):
        pass

    def release(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class _SameThreadFuture(object):
    def __init__(self, id, result):
        self._result = result
        self.id = id
        self._condition = _FakeCondition()
        self._state = FINISHED
        self._waiters = []

    def result(self):
        return self._result


class SameThreadExecutor(Executor):
    """A mock executor. Guarantees stable order of future evaluations: a submitted future is
    evaluated immediately as part of submit implementation."""

    def __init__(self):
        # A future has to have a unique id. Use this counter to generate the id.
        self._id = 0

    def submit(self, fn, *args, **kwargs):
        self._id += 1
        return _SameThreadFuture(self._id, fn(*args, **kwargs))
