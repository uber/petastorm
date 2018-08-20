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
from concurrent.futures._base import Future


class SameThreadExecutor(Executor):
    """A mock executor. Guarantees stable order of future evaluations: a submitted future is
    evaluated immediately as part of submit implementation."""

    def submit(self, fn, *args, **kwargs):
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future
