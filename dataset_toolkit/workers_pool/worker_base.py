#
# Uber, Inc. (c) 2017
#
from abc import abstractmethod


class WorkerBase(object):
    def __init__(self, worker_id, publish_func, args):
        """Initializes a worker.

        :param worker_id: An integer uniquely identifying a worker instance
        :param publish_func: Function handler to be used to publish data
        :param args: application specific args
        """
        self.worker_id = worker_id
        self.publish_func = publish_func
        self.args = args

    @abstractmethod
    def process(self, *args, **kargs):
        pass

    def shutdown(self):
        pass
