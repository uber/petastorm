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

from time import sleep

from petastorm.workers_pool.worker_base import WorkerBase


class CoeffMultiplierWorker(WorkerBase):
    def process(self, *args, **kargs):
        # If value is a list, generate multiple outputs
        value = kargs['value']
        if isinstance(value, list):
            for v in value:
                self.publish_func(v * self.args['coeff'])
        else:
            self.publish_func(value * self.args['coeff'])


class IdentityWorker(WorkerBase):
    def process(self, *args, **kargs):
        self.publish_func(kargs['item'])


class WorkerIdGeneratingWorker(WorkerBase):
    def process(self, *args, **kargs):
        self.publish_func(self.worker_id)


class WorkerMultiIdGeneratingWorker(WorkerBase):
    def process(self, *args, **kargs):
        for _ in range(2):
            self.publish_func(self.worker_id)


class SleepyDoingNothingWorker(WorkerIdGeneratingWorker):
    def __init__(self, worker_id, publish_func, args):
        """
        :param args: The worker will sleep for this time (seconds) until returned
        """
        super(SleepyDoingNothingWorker, self).__init__(worker_id, publish_func, args)
        self._sleep = args

    def process(self, *args, **kargs):
        sleep(self._sleep)


class SleepyWorkerIdGeneratingWorker(WorkerIdGeneratingWorker):
    def process(self, *args, **kargs):
        sleep(1)
        super(SleepyWorkerIdGeneratingWorker, self).process()


class ExceptionGeneratingWorker_5(WorkerBase):
    def process(self, *args, **kargs):
        raise ValueError("worker %d raise test exception - IT SHOULD BE EXCEPTION!" % self.worker_id)


class PreprogrammedReturnValueWorker(WorkerBase):
    def __init__(self, worker_id, publish_func, args):
        """
        :param args: Array of arrays. Defines which values to return at consequent process calls. For example,
        args = [[], [1], [12, 13]] will result in process not generating any results in the first call, generating '1'
        in the second and '12', '13' at the third invocation of 'process'
        """
        super(PreprogrammedReturnValueWorker, self).__init__(worker_id, publish_func, args)
        self._program = args
        self._current_step = 0

    def process(self, *args, **kargs):
        for value in self._program[self._current_step]:
            self.publish_func(value)
        self._current_step += 1
