#
# Uber, Inc. (c) 2018
#
import abc


class CacheBase(object):
    @abc.abstractmethod
    def get(self, key, fill_cache_func):
        """Gets an entry from the cache implementation.

        If there is a cache miss, fill_cache_func() will be evaluated to get the value.

        :param key: A key identifying cache entry
        :param fill_cache_func: This function will be evaluated (fill_cache_func()) to populate cache, if the no
        value is present in the cache.
        :return: A value from cache
        """
        pass


class NullCache(CacheBase):
    """A pass-through cache implementation: value generating function will be called each """

    def get(self, key, fill_cache_func):
        return fill_cache_func()
