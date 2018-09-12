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
from torch.utils.data.dataloader import default_collate


class DataLoader(object):
    """
    A data loader adaptor for ``torch.utils.data.DataLoader``.

    This class iterates and returns items from the Reader in batches.

    This loader can be used as a context manager, but it will terminate at the end of an epoch.
    The context will invoke next_epoch() upon entry.

    If not used as context manager, invoke the next_epoch() function at the start of each epoch, and
    once more at the very end.
    """

    def __init__(self, reader, batch_size=1, collate_fn=default_collate, transform=None):
        """
        Initializes a data loader object, with a default collate and optional transform functions.

        This loader handles multiple epochs by instantiating a new Reader per epoch.

        :param reader: petastorm Reader instance
        :param batch_size: the number of items to return per batch; factored into the len() of this reader
        :param collate_fn: a optional callable to merge a list of samples to form a mini-batch.
        :param transform: a optional tranform function to apply to each data row
        """
        self.reader = reader
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.transform = transform

    def __iter__(self):
        """
        The Data Loader iterator stops the for-loop at the end of each epoch, but a subsequent for-loop
        will instantiate a new Reader and yield more results, until the requested number of epoch has been
        reached.  After that point, any subsequent call results in StopIteration, per iterator protocol.
        """
        batch = []
        for row in self.reader:
            # Default collate does not work nicely on namedtuples and treat them as lists
            # Using dict will result in the yielded structures being dicts as well
            row_as_dict = row._asdict()
            batch.append(self.transform(row_as_dict) if self.transform else row_as_dict)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if batch:
            yield self.collate_fn(batch)

    # Functions needed to treat data loader as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.stop()
        self.reader.join()
