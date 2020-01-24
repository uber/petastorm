import logging
import zmq
import sys
from psutil import process_iter
from time import sleep

from petastorm.workers_pool.constants import LOCALHOST, SOCKET_LINGER_MS, CONTROL_FINISHED

logger = logging.getLogger(__name__)


class ProcessMonitor(object):

    def __init__(self, address=None):
        self.address = address

    def bind(self):
        logger.debug('Connecting sockets')
        self._context = zmq.Context()
        self._publisher = self._context.socket(zmq.PUB)
        if self.address:
            self._publisher.bind(self.address)
        else:
            self._port = self._publisher.bind_to_random_port(LOCALHOST)
            self.address = self._get_address()

    def unbind(self):
        self._publisher.unbind(self.address)
        self._publisher.close()
        self._context.destroy()

    def _get_address(self):
        return '{}:{}'.format(LOCALHOST, self._port)

    @staticmethod
    def bootstrap(main_process_pid, process_monitor_address, control_address, polling_time=2):
        logger.debug('Starting process_monitor_bootstrap')

        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        publisher.linger = SOCKET_LINGER_MS
        publisher.bind(process_monitor_address)

        main_thread_receiver = context.socket(zmq.SUB)
        main_thread_receiver.connect(control_address)
        main_thread_receiver.setsockopt(zmq.SUBSCRIBE, b"")
        poller = zmq.Poller()
        poller.register(main_thread_receiver, zmq.POLLIN)

        while True:
            sleep(polling_time)
            socks = dict(poller.poll(1000))
            if socks.get(main_thread_receiver) == zmq.POLLIN:
                control_message = main_thread_receiver.recv_string()
                if control_message == CONTROL_FINISHED:
                    sys.stderr.write('Got a stop message from main thread. Exiting\n')
                    publisher.close()
                    main_thread_receiver.close()
                    context.destroy()
                    break

            main_process_is_dead = (main_process_pid not in
                                    [process.pid for process in process_iter()
                                     if process.status() != 'zombie'])
            if main_process_is_dead:
                sys.stderr.write("Main process with pid: %d is dead. "
                                 "Publishing %s to petastorm workers on %s\n"
                                 % (main_process_pid, CONTROL_FINISHED, process_monitor_address))
                publisher.send_string(CONTROL_FINISHED)
                publisher.close()
                main_thread_receiver.close()
                context.destroy()
                sys.stderr.write('Process monitor for pid: %d exiting\n' % main_process_pid)
                break
