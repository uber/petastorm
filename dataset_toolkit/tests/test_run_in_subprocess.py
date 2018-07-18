import unittest
from functools import partial

from dataset_toolkit.unischema import dict_to_spark_row, Unischema
from dataset_toolkit.utils import run_in_subprocess


def builtin_func():
    return range(10)


def multiply(a, b):
    return a * b


class RunInSubprocessTest(unittest.TestCase):

    def test_run_in_subprocess(self):
        # Serialization of a built in function
        self.assertEquals(run_in_subprocess(builtin_func), builtin_func())

        # Arg passing
        self.assertEquals(run_in_subprocess(multiply, 2, 3), 6)

    def test_partial_application(self):
        unischema = Unischema('foo', [])
        func = partial(dict_to_spark_row, unischema)
        func({})

        # Must pass as positional arg in the right order
        func = partial(dict_to_spark_row, {})
        with self.assertRaises(AssertionError):
            func(Unischema)


if __name__ == '__main__':
    unittest.main()
