.. inclusion-marker-start-do-not-remove

==================
To petastorm 0.5.0
==================

Petastorm 0.5.0 has some breaking changes from previous versions. These include:

- Users should use :func:`~petastorm.reader.make_reader`, instead of instantiating :class:`~petastorm.reader.Reader`
  directly to create a new instances
- It is still possible (although discouraged in most cases) to instantitate :class:`~petastorm.reader.Reader`. Some of
  its argument has changed.

Use :func:`~petastorm.reader.make_reader` to instantiate a reader instance
--------------------------------------------------------------------------

Use :func:`~petastorm.reader.make_reader` to create a new instance of a reader. :func:`~petastorm.reader.make_reader`
takes arguments that are almost similar to constructor arguments of :class:`~petastorm.reader.Reader`. The following
list enumerates the differences:

- ``reader_pool_type``: takes one of the strings: ``'thread'``, ``'process'``, ``'dummy'``
  (instead of ``ThreadPool()``, ``ProcessPool()`` and ``DummyPool()`` object instances). Pass number of workers using
  ``workers_count`` argument.
- ``training_partition`` and ``num_training_partitions`` were renamed into ``cur_shard`` and ``shard_count``.
- ``shuffle`` and ``shuffle_options`` were replaced by ``shuffle_row_groups=True, shuffle_row_drop_partitions=1``

.. code-block:: python

    from petastorm.reader import Reader
    reader = Reader(dataset_url,
                    reader_pool=ThreadPool(5),
                    training_partition=1, num_training_partitions=5,
                    shuffle_options=ShuffleOptions(shuffle_row_groups=False))

To:

.. code-block:: python

    from petastorm import make_reader
    reader = make_reader(dataset_url,
                         reader_pool_type='thread',
                         workers_count=5,
                         cur_shard=1, shard_count=5,
                         shuffle_row_groups=False)

