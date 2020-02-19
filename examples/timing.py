import time
import numpy as np
from petastorm.pytorch import DataLoader
from collections import namedtuple
import torch
from functools import partial
import sys


class DummyReader(object):
    @property
    def is_batched_reader(self):
        return True

    def stop(self):
        pass

    def join(self):
        pass

    def __iter__(self):
        nt = namedtuple("row", ["test"])
        batch = nt(np.random.rand(1000, 64).astype(np.float32))
        while True:
            yield batch


def main(device):
    print("Testing DataLoader on", device)
    reader = DummyReader()
    for batch_size in [10, 100, 1000, 100000]:
        iterations = 100
        loader = DataLoader(reader, shuffling_queue_capacity=batch_size * 10, batch_size=batch_size, collate_fn=partial(torch.as_tensor, device=device))
        it = iter(loader)

        # Warmup
        for _ in range(iterations):
            next(it)
        print("Done warming up")

        tstart = time.time()
        for _ in range(iterations):
            next(it)
        print("Samples per second for batch {}: {:.4g}".format(batch_size, (iterations * batch_size) / (time.time() - tstart)))


if __name__ == "__main__":
    main(sys.argv[-1] if len(sys.argv) > 1 else "cpu")
