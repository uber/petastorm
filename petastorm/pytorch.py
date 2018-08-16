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

from petastorm.reader import Reader, ShuffleOptions
from petastorm.workers_pool.thread_pool import ThreadPool

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

    def __init__(self, dataset_url, schema_fields=None, batch_size=1, shuffle_options=None,
                 collate_fn=default_collate, transform=None,
                 predicate=None, rowgroup_selector=None,
                 num_workers=0, reader_pool=None, num_epochs=1, sequence=None,
                 training_partition=None, num_training_partitions=None,
                 read_timeout_s=None, cache=None):
        """
        Initializes a batch reader object, with a default collate and optional transform functions.

        This loader handles multiple epochs by instantiating a new Reader per epoch.

        :param dataset_url: an filepath or a url to a parquet directory,
                       e.g. 'hdfs://some_hdfs_cluster/user/yevgeni/parquet8', or '/tmp/mydataset'
        :param schema_fields: list of unischema fields to subset, or None to read all fields.
        :param batch_size: the number of items to return per batch; factored into the len() of this reader
        :param shuffle_options : ShuffleOptions object to describe how to shuffle dataset (supercedes shuffle parameter)
                       defaults to shuffling row groups but not to drop rows based on partitions.
                       Set to True for default shuffle options.
        :param collate_fn: a optional callable to merge a list of samples to form a mini-batch.
        :param transform: a optional tranform function to apply to each data row

        :param predicate: instance of predicate object to filter rows to be returned by reader.
        :param rowgroup_selector: instance of row group selector object to select row groups to be read
        :param num_workers: how many subprocesses to use for data loading; affects the reader_pool option.
                       0 means that the data will be loaded using a single thread (default: 0)
        :param reader_pool: parallelization pool.
                       This pool is a custom implementation used to parallelize reading data from the dataset.
                       Any object from workers_pool package can be used (e.g. ProcessPool)
        :param num_epochs: An epoch is a single pass over all samples in the dataset. Setting num_epochs to 'None' will
                       result in an infinite number of epochs.
        :param sequence: If it is set to a Sequence object, then fetch will return a sequence, otherwise fetch
                       will return an item.
        :param training_partition: An int denoting the partition number used for multi node training. Each node should
                       pass in a unique partition number in the range [0, num_training_partitions).
                       num_training_partitions must be supplied as well.
        :param num_training_partitions An int denoting the number of training partitions (how many nodes are performing
                       the multi node training)
        :param read_timeout_s: A numeric with the amount of time in seconds you would like to give a read before it
                       times out and raises an EmptyResultError. Pass in None for an infinite timeout
        :param cache: An object conforming to `cache.CacheBase` interface. Before loading row groups from a parquet file
                       the Reader will attempt to load these values from cache. Caching is useful when communication
                       to the main data store is either slow or expensive and the local machine has large enough storage
                       to store entire dataset (or a partition of a dataset if num_training_partitions is used).
        """
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.num_epochs = num_epochs
        self.transform = transform

        if num_workers == 0:
            # Use 1 thread
            num_workers = 1
        if shuffle_options == True:
            shuffle_options = ShuffleOptions()
        self.reader = Reader(dataset_url, schema_fields=schema_fields,
                             predicate=predicate, rowgroup_selector=rowgroup_selector,
                             reader_pool=ThreadPool(num_workers), num_epochs=num_epochs, sequence=sequence,
                             training_partition=training_partition, num_training_partitions=num_training_partitions,
                             read_timeout_s=read_timeout_s, cache=cache, shuffle_options=shuffle_options)

    def __len__(self):
        return (len(self.reader) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """
        The Data Loader iterator stops the for-loop at the end of each epoch, but a subsequent for-loop
        will instantiate a new Reader and yield more results, until the requested number of epoch has been
        reached.  After that point, any subsequent call results in StopIteration, per iterator protocol.
        """
        batch = []
        for row in self.reader:
            batch.append(self.transform(row))
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
