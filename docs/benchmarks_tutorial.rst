Benchmarks
==========

``petastorm-throughput.py`` command line tool makes it possible to measure :class:`~petastorm.reader.Reader` samples
throughtput for a given dataset. The command line tool has several optional command line arguments, such as number of
threads being used by the :class:`~petastorm.reader.Reader`, whether to use `Tensorflow` way of reading data
(:func:`~petastorm.tf_utils.tf_tensors`) or pure Python one.

As a toy example, let's generate a small `Hello World` dataset (by default the dataset is generated here:
``/tmp/hello_world_dataset``):

.. code-block:: bash

 $ python examples/hello_world/generate_hello_world_dataset.py

Now, let's run ``petastorm-throughput``:

.. code-block:: bash

 $ petastorm-throughput.py file:///tmp/hello_world_dataset
 Average sample read rate: 709.84 samples/sec; RAM 217.12 MB (rss); CPU 136.30%

Typically, you would want to find an optimal throughput for your system by varying the following parameters:

 - ``-w``: number of workers the reader is using to load and decode data. Could be either number of threads or processes, depends on the ``-p`` argument.
 - ``-p``: thread or process based parallelism.


Before actual throughput measurement is conducted, a number of warmup readings are executed to bring the system into a
steady state. The amount of warmup reads as well as measurements reads are controlled by ``-m`` and ``-n`` command line
arguments respectively. The following example sets the number of warmup reads to a ``1000``` and measurement reads to
``5000``:

.. code-block:: bash

 $ petastorm-throughput.py file:///tmp/hello_world_dataset -m 1000 -n 5000
 Average sample read rate: 653.10 samples/sec; RAM 219.86 MB (rss); CPU 131.50%
