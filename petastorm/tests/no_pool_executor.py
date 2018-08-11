from concurrent.futures import Executor
from concurrent.futures._base import FINISHED


class FakeCondition(object):
    def acquire(self):
        pass
    def release(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class NoPoolFuture(object):
    def __init__(self, id, result):
        self._result = result
        self.id = id
        self._condition = FakeCondition()
        self._state = FINISHED
        self._waiters = []



    def result(self):
        return self._result





class NoPoolExecutor(Executor):
    def __init__(self):
        self._id = 0
        pass

    def submit(self, fn, *args, **kwargs):
        self._id += 1
        return NoPoolFuture(self._id, fn(*args, **kwargs))
