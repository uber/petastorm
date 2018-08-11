from time import sleep

import numpy as np
import six
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from six.moves.queue import Queue


def ventilator():
    return iter(range(100))


def load(rowgroup_spec):
    s = np.random.random() * 0.3
    print('Loading:' + str(rowgroup_spec))
    sleep(s)
    return rowgroup_spec


def decode(rowgroup_spec):
    s = np.random.random() * 0.3
    print('Decoding:' + str(rowgroup_spec))
    sleep(s)
    return rowgroup_spec


class ShufflingQueue(object):
    def __init__(self):
        self.store = []

    def extend(self, data):
        # print('shuffling in : ' + str(data))
        self.store.extend(data)

    def pull(self):
        return self.store.pop(0)

    def has_more_to_offer(self):
        return len(self.store) > 0


class EOFSentinel(object):
    pass


# Runs in a separate thread
def flow_manager_loop(epochs_generator, loading_pool, loader, decoding_pool, decoder, shuffling_queue, output_queue,
                      stop_event):
    # Possibly infinite loop driven by row-group ventilation logic (infinite if epochs=inf)
    in_loading = []
    in_decoding = []

    try:
        epochs_iterator = iter(epochs_generator())
        rowgroup_spec = next(epochs_iterator)
        while rowgroup_spec is not None or in_loading or in_decoding:
            if stop_event.is_set():
                break
            # print('1')
            # Submit next task
            if rowgroup_spec is not None and len(in_loading) < 20:
                # print 'Submitted ' + str(rowgroup_spec)
                in_loading.append(loading_pool.submit(loader.load, rowgroup_spec))

                try:
                    rowgroup_spec = next(epochs_iterator)
                except StopIteration:
                    rowgroup_spec = None

            loaded = as_completed(in_loading, timeout=0)

            # print('2')
            # Push all loaded and partially decoded data into shuffling queue
            while True:
                try:
                    curr_loaded = next(loaded)
                    # print 'Loaded ' + str(curr_loaded.result()[0]['id']) + ' ' + str(curr_loaded.result()[-1]['id'])

                    shuffling_queue.extend(curr_loaded.result())
                    in_loading.remove(curr_loaded)
                except TimeoutError:
                    break
                except StopIteration:
                    break

            # print('3')

            # Pump from the shuffling queue (to the decoding pool
            move_count = 0
            while shuffling_queue.has_more_to_offer():
                # print('3.1', len(in_loading), len(in_decoding), output_queue.qsize())
                shuffled = shuffling_queue.pull()
                # print 'Pulled shuffled ' + str(shuffled['id'] )

                # print('3.2', len(in_loading), len(in_decoding), output_queue.qsize())
                in_decoding.append(decoding_pool.submit(decoder, shuffled))

                move_count += 1
                if move_count > 40: break

            # print('4', len(in_loading), len(in_decoding), output_queue.qsize())

            decoded = as_completed(in_decoding, timeout=0)
            already_decoded = set()
            while True:
                try:
                    already_decoded.add(next(decoded))
                except TimeoutError:
                    break
                except StopIteration:
                    break

            # print('5', len(in_loading), len(in_decoding), output_queue.qsize())

            for i in six.moves.xrange(len(in_decoding) - 1, -1, -1):
                was_decoding = in_decoding[i]
                if was_decoding in already_decoded:
                    # print 'Writing to output queue' + str(was_decoding.result().id)

                    output_queue.put(was_decoding.result())
                    # TODO(yevgeni): remove is linear time complexity. Can do better?
                    del in_decoding[i]

            # print('6')

            print(len(in_loading), len(in_decoding), output_queue.qsize())

        output_queue.put(EOFSentinel())

    except Exception as e:
        output_queue.put(e)


if __name__ == '__main__':
    output_queue = Queue()
    flow_manager_loop(ventilator(), ThreadPoolExecutor(2), ThreadPoolExecutor(2), ShufflingQueue(), output_queue)

    print('Done:')
    while output_queue.qsize() > 0:
        print(output_queue.get_nowait())
