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

import dill
from concurrent.futures import ProcessPoolExecutor


def _decode_run(payload):
    fun, args, kwargs = dill.loads(payload)
    return fun(*args, **kwargs)


def submit_with_dill(executor, fun, *args, **kwargs):
    """Submits a function on a `concurrent.futures` executors using `dill` serialization.

    `dill` is better in serializing code. This is handy when submitting lambda's on an ProcessPoolExecutor.

    Will only use dill serialization for `ProcessPoolExecutor` (explicitly checks the type of the executor) to avoid
    unnecessary serlialization for in-process pool executors.

    Example:
      e = ProcessPoolExecutor(10)
      submit_with_dill(e, lambda x: 2 * x, 10)

    :param executor: An instance of a concurrent.futures Executor object.
    :param fun: Function object to submit
    :param args: parameters passed to fun
    :param kwargs: named parameters passed to fun
    :return: A future
    """
    # Save serialization/deserialization for in-process executors.
    if isinstance(executor, ProcessPoolExecutor):
        payload = dill.dumps((fun, args, kwargs))
        return executor.submit(_decode_run, payload)
    else:
        return executor.submit(fun, *args, **kwargs)
