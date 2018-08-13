class ShufflingQueue(object):
    def __init__(self):
        self.store = []

    def extend(self, data):
        self.store.extend(data)

    def pull(self):
        return self.store.pop(0)

    def has_more_to_offer(self):
        return len(self.store) > 0