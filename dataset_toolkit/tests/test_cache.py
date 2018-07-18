#
# Uber, Inc. (c) 2018
#
import unittest

from dataset_toolkit.cache import NullCache


class TestNullCache(unittest.TestCase):

    def test_null_cache(self):
        """Testing trivial NullCache: should trigger value generating function on each run"""
        cache = NullCache()
        self.assertEqual(42, cache.get('some_key', lambda: 42))


if __name__ == '__main__':
    # Delegate to the test framework.
    unittest.main()
