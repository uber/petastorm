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


import time
import unittest

from petastorm.workers_pool import EmptyResultError
from petastorm.workers_pool.dummy_pool import DummyPool
from petastorm.workers_pool.process_pool import ProcessPool
from petastorm.workers_pool.tests.stub_workers import IdentityWorker
from petastorm.workers_pool.thread_pool import ThreadPool
from petastorm.workers_pool.ventilator import ConcurrentVentilator


class TestWorkersPool(unittest.TestCase):

    def _test_simple_ventilation(self, pool_class_factory):
        pool = pool_class_factory()

        items_to_ventilate = [{'item': i} for i in range(50)]
        ventilator = ConcurrentVentilator(ventilate_fn=pool.ventilate, items_to_ventilate=items_to_ventilate)
        pool.start(IdentityWorker, ventilator=ventilator)

        all_results = [pool.get_results() for _ in items_to_ventilate]
        self.assertEqual([i['item'] for i in items_to_ventilate], sorted(all_results))

        pool.stop()
        pool.join()

    def test_ventilator_processes(self):
        self._test_simple_ventilation(lambda: ProcessPool(10))

    def test_ventilator_threads(self):
        self._test_simple_ventilation(lambda: ThreadPool(10))

    def test_ventilator_dummy(self):
        self._test_simple_ventilation(DummyPool)

    def test_max_ventilation_size(self):
        """Tests that we dont surpass a max ventilation size in each pool type
        (since it relies on accurate ventilation size reporting)"""
        max_ventilation_size = 10

        for pool in [DummyPool(), ProcessPool(10), ThreadPool(10)]:
            ventilator = ConcurrentVentilator(ventilate_fn=pool.ventilate,
                                              items_to_ventilate=[{'item': i} for i in range(100)],
                                              max_ventilation_queue_size=max_ventilation_size)
            pool.start(IdentityWorker, ventilator=ventilator)

            # Give time for the thread to fill the ventilation queue
            while ventilator._ventilated_items_count - ventilator._processed_items_count < max_ventilation_size:
                time.sleep(.1)

            # After stopping the ventilator queue, we should only get 10 results
            ventilator.stop()
            for _ in range(max_ventilation_size):
                pool.get_results()

            with self.assertRaises(EmptyResultError):
                pool.get_results()

            pool.stop()
            pool.join()

    def test_reset_in_the_middle_of_ventilation(self):
        """Can not reset ventilator in the middle of ventilation"""
        for pool in [DummyPool(), ThreadPool(10)]:
            ventilator = ConcurrentVentilator(ventilate_fn=pool.ventilate,
                                              items_to_ventilate=[{'item': i} for i in range(100)],
                                              iterations=None)
            pool.start(IdentityWorker, ventilator=ventilator)

            # Resetting is supported only when the ventilator has finished
            with self.assertRaises(NotImplementedError):
                ventilator.reset()

            pool.stop()
            pool.join()

    def test_reset_ventilator(self):
        """Resetting ventilator after all items were ventilated will make it re-ventilate the same items"""
        items_count = 100
        for pool in [DummyPool(), ThreadPool(10)]:
            ventilator = ConcurrentVentilator(ventilate_fn=pool.ventilate,
                                              items_to_ventilate=[{'item': i} for i in range(items_count)],
                                              iterations=1)
            pool.start(IdentityWorker, ventilator=ventilator)

            # Readout all ventilated items
            for _ in range(items_count):
                pool.get_results()

            # Should fail reading the next, as all items were read by now
            with self.assertRaises(EmptyResultError):
                pool.get_results()

            # Resetting, hence will be read out the items all over again
            ventilator.reset()

            for _ in range(items_count):
                pool.get_results()

            with self.assertRaises(EmptyResultError):
                pool.get_results()

            pool.stop()
            pool.join()

    def test_empty_ventilation(self):
        pool = DummyPool()
        ventilator = ConcurrentVentilator(pool.ventilate, [])
        pool.start(IdentityWorker, ventilator=ventilator)
        with self.assertRaises(EmptyResultError):
            pool.get_results()

        pool.stop()
        pool.join()

    def test_multiple_iterations(self):
        size = 10
        iterations = 5

        pool = DummyPool()
        ventilator = ConcurrentVentilator(pool.ventilate, [{'item': i} for i in range(size)], iterations=iterations)
        pool.start(IdentityWorker, ventilator=ventilator)

        results = [pool.get_results() for _ in range(size * iterations)]
        self.assertEqual(sorted(results), sorted(list(range(size)) * iterations))
        with self.assertRaises(EmptyResultError):
            pool.get_results()

        ventilator.stop()
        pool.stop()
        pool.join()

    def test_ventilator_stop(self):
        size = 100
        max_ventilation_queue_size = 10

        pool = DummyPool()

        ventilator = ConcurrentVentilator(ventilate_fn=pool.ventilate,
                                          items_to_ventilate=[{'item': i} for i in range(size)],
                                          max_ventilation_queue_size=max_ventilation_queue_size)
        pool.start(IdentityWorker, ventilator=ventilator)

        [pool.get_results() for _ in range(max_ventilation_queue_size)]

        # Stop the ventilator queue after some time, so there should only be 10 items left on it
        while ventilator._ventilated_items_count - ventilator._processed_items_count < max_ventilation_queue_size:
            time.sleep(.1)

        ventilator.stop()

        [pool.get_results() for _ in range(max_ventilation_queue_size)]
        with self.assertRaises(EmptyResultError):
            pool.get_results()

        pool.stop()
        pool.join()

    def test_randomize_item_order(self):
        size = 100
        pool = DummyPool()
        items_to_ventilate_1 = [{'item': i} for i in range(size)]
        ventilator = ConcurrentVentilator(pool.ventilate, items_to_ventilate=items_to_ventilate_1)
        pool.start(IdentityWorker, ventilator=ventilator)
        first_results = [pool.get_results() for _ in range(size)]
        pool.stop()
        pool.join()

        pool = DummyPool()
        items_to_ventilate_2 = [{'item': i} for i in range(size)]
        ventilator = ConcurrentVentilator(pool.ventilate,
                                          items_to_ventilate=items_to_ventilate_2,
                                          randomize_item_order=True)
        pool.start(IdentityWorker, ventilator=ventilator)
        second_results = [pool.get_results() for _ in range(size)]
        pool.stop()
        pool.join()

        # Because we're using the dummy pool, without randomizing item order the results
        # should be exactly the list of ventilation items
        self.assertEqual(first_results, [item['item'] for item in items_to_ventilate_1])
        self.assertNotEqual(first_results, second_results)

        pool.stop()
        pool.join()


if __name__ == '__main__':
    # Delegate to the test framework.
    unittest.main()
