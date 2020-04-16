#  Copyright (c) 2017-2020 Uber Technologies, Inc.
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
from __future__ import print_function
from __future__ import division
import time
import numpy as np
from petastorm.pytorch import BatchedDataLoader, DataLoader
from collections import namedtuple
import torch
from functools import partial
import sys


class DummyReader(object):
    def __init__(self, batch=1000, dim=64):
        self.batch = batch
        self.dim = dim

    @property
    def is_batched_reader(self):
        return True

    def stop(self):
        pass

    def join(self):
        pass

    def __iter__(self):
        nt = namedtuple("row", ["test"])
        batch = nt(np.random.rand(self.batch, self.dim).astype(np.float32))
        while True:
            yield batch


def main(device='cpu', batch=1000, dim=64):
    print("Testing DataLoader on cpu")
    reader = DummyReader(int(batch), int(dim))

    for batch_size in [10, 100, 1000]:
        iterations = 100
        loader = DataLoader(reader, shuffling_queue_capacity=batch_size * 10, batch_size=batch_size)
        it = iter(loader)

        # Warmup
        for _ in range(iterations):
            next(it)
        print("Done warming up")

        tstart = time.time()
        for _ in range(iterations):
            next(it)
        print("Samples per second for batch {}: {:.4g}".format(
            batch_size, (iterations * batch_size) / (time.time() - tstart)))

    print("Testing BatchedDataLoader on", device)
    for batch_size in [10, 100, 1000, 100000]:
        iterations = 100
        loader = BatchedDataLoader(reader, shuffling_queue_capacity=batch_size * 10, batch_size=batch_size,
                                   transform_fn=partial(torch.as_tensor, device=device))
        it = iter(loader)

        # Warmup
        for _ in range(iterations):
            next(it)
        print("Done warming up")

        tstart = time.time()
        for _ in range(iterations):
            next(it)
        print("Samples per second for batch {}: {:.4g}".format(
            batch_size, (iterations * batch_size) / (time.time() - tstart)))


if __name__ == "__main__":
    main(*sys.argv[1:])
