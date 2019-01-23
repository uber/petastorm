import abc
from contextlib import contextmanager
from time import time

import six


@six.add_metaclass(abc.ABCMeta)
class StatisticsCollector(object):
    """ Base Class For Collecting And Emitting Statistics On The Petastorm Reader """

    @abc.abstractmethod
    @contextmanager
    def measure_row_retrieval(self):
        """Count the number of rows (or batches in batch reader) retrieved by the client and times how long it takes"""
        pass

    @abc.abstractmethod
    @contextmanager
    def measure_reader_startup(self):
        """Time how long it takes for the petastorm reader to be instantiated"""
        pass

    @abc.abstractmethod
    @contextmanager
    def measure_reader_join(self):
        """Time how long it takes for the petastorm reader threads to be joined"""
        pass

    @abc.abstractmethod
    @contextmanager
    def measure_reader_stop(self):
        """Time how long it takes for the petastorm reader to be shutdown"""
        pass


class InMemoryStatisticsCollector(StatisticsCollector):

    rows_retrieved = 0
    row_retrieval_times = []
    time_startup = None
    time_join = None
    time_stop = None

    @contextmanager
    def measure_row_retrieval(self):
        tic = time()
        yield
        toc = time()
        self.row_retrieval_times.append(toc - tic)
        self.rows_retrieved += 1

    @contextmanager
    def measure_reader_startup(self):
        tic = time()
        yield
        self.time_startup = time() - tic

    @contextmanager
    def measure_reader_join(self):
        tic = time()
        yield
        self.time_join = time() - tic

    @contextmanager
    def measure_reader_stop(self):
        tic = time()
        yield
        self.time_stop = time() - tic


class NoopStatisticsCollector(StatisticsCollector):

    def measure_row_retrieval(self):
        pass

    def measure_reader_startup(self):
        pass

    def measure_reader_join(self):
        pass

    def measure_reader_stop(self):
        pass
