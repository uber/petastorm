#
# Uber, Inc. (c) 2017
#


class EmptyResultError(RuntimeError):
    """Exception used to signal that there are no new elements in the queue and no new elements are expected, unless
    ventilate is called again"""
    pass


class TimeoutWaitingForResultError(RuntimeError):
    """Indicates that timeout has elapsed while waiting for a result"""
    pass


class VentilatedItemProcessedMessage(object):
    """Object to signal that a worker has completed processing an item from the ventilation queue"""
    pass
