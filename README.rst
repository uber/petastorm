=========
Petastorm
=========

.. image:: https://travis-ci.com/uber/petastorm.svg?branch=master
   :target: https://travis-ci.com/uber/petastorm
   :alt: Build Status (Travis CI)

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :alt: License

.. image:: https://badge.fury.io/py/petastorm.svg
   :target: https://pypi.org/project/petastorm
   :alt: Latest Version

.. _Tensorflow: http://www.tensorflow.org/
.. _PyTorch: https://pytorch.org/
.. _PySpark: http://spark.apache.org/docs/latest/api/python/pyspark.html


.. inclusion-marker-start-do-not-remove

Petastorm, an open source data access library developed at Uber ATG. This library enables single machine or
distributed training and evaluation of deep learning models directly from multi-terabyte datasets in Apache Parquet
format. Petastorm supports popular Python-based machine learning (ML) frameworks such as Tensorflow_,
Pytorch_, and PySpark_. It can also be used from pure Python code.


Install
-------

.. code-block:: bash

    pip install petastorm


There are several extra dependencies that are defined by the ``petastorm`` package that are not installed automatically.
The extras are: ``tf``, ``tf_gpu``, ``torch``, ``open_cv``, ``docs``, ``tests``.

For example to trigger installation of GPU version of tensorflow and opencv, use the following pip command:

.. code-block:: bash

    pip install petastorm[opencv,tf_gpu]


Usage
-----

Generating a dataset
^^^^^^^^^^^^^^^^^^^^

A dataset created using Petastorm is stored in `Apache Parquet <https://parquet.apache.org/>`_ format.
On top of a Parquet schema, petastorm also stores higher-level schema information that makes multidimensional arrays into a native part of a petastorm dataset. 

Petastorm supports extensible data codecs. These enable a user to use one of the standard data compressions (jpeg, png) or implement her own.

Generating a dataset is done using PySpark.
PySpark natively supports Parquet format, making it easy to run on a single machine or on a Spark compute cluster.
Here is a minimalistic example writing out a table with some random data.


.. code-block:: python

    HelloWorldSchema = Unischema('HelloWorldSchema', [
       UnischemaField('id', np.int32, (), ScalarCodec(IntegerType()), False),
       UnischemaField('image1', np.uint8, (128, 256, 3), CompressedImageCodec('png'), False),
       UnischemaField('other_data', np.uint8, (None, 128, 30, None), NdarrayCodec(), False),
    ])


    def row_generator(x):
       """Returns a single entry in the generated dataset. Return a bunch of random values as an example."""
       return {'id': x,
               'image1': np.random.randint(0, 255, dtype=np.uint8, size=(128, 256, 3)),
               'other_data': np.random.randint(0, 255, dtype=np.uint8, size=(4, 128, 30, 3))}


    def generate_hello_world_dataset(output_url='file:///tmp/hello_world_dataset'):
       rows_count = 10
       rowgroup_size_mb = 256

       spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()
       sc = spark.sparkContext

       # Wrap dataset materialization portion. Will take care of setting up spark environment variables as
       # well as save petastorm specific metadata
       with materialize_dataset(spark, output_url, HelloWorldSchema, rowgroup_size_mb):

           rows_rdd = sc.parallelize(range(rows_count))\
               .map(row_generator)\
               .map(lambda x: dict_to_spark_row(HelloWorldSchema, x))

           spark.createDataFrame(rows_rdd, HelloWorldSchema.as_spark_schema()) \
               .coalesce(10) \
               .write \
               .mode('overwrite') \
               .parquet(output_url)

- ``HelloWorldSchema`` is an instance of a ``Unischema`` object.
  ``Unischema`` is capable of rendering types of its fields into different
  framework specific formats, such as: Spark’s ``StructType``, Tensorflow’s
  ``tf.DType`` and numpy’s ``numpy.dtype``.
- To define a dataset field, you need to specify a ``type``, ``shape``, a
  ``codec`` instance and whether the field is nullable for each field of the
  ``Unischema``.
- We use PySpark for writing output Parquet files. In this example, we launch
  PySpark on a local box (``.master('local[2]')``). Of course for a larger
  scale dataset generation we would need a real compute cluster.
- We wrap spark dataset generation code with the ``materialize_dataset``
  context manager.  The context manager is responsible for configuring row
  group size at the beginning and write out petastorm specific metadata at the
  end.
- The row generating code is expected to return a Python dictionary indexed by
  a field name. We use ``row_generator`` function for that. 
- ``dict_to_spark_row`` converts the dictionary into a ``pyspark.Row``
  object while ensuring schema ``HelloWorldSchema`` compliance (shape,
  type and is-nullable condition are tested).
- Once we have a ``pyspark.DataFrame`` we write it out to a parquet
  storage. The parquet schema is automatically derived from
  ``HelloWorldSchema``.

Reading a dataset from Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reading a dataset is simple using the ``petastorm.reader.Reader`` class:

.. code-block:: python

    with Reader('hdfs://myhadoop/some_dataset') as reader:
       for row in reader:
           print(row)

``Reader`` supports taking either an ``hdfs://...`` or ``file://...``
protocol URI.

Once a ``Reader`` is instantiated, you can use it as an iterator.

Reading a dataset using Tensorflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To hookup the reader into a tensorflow graph, you can use the ``tf_tensors``
function:

.. code-block:: python

    with Reader('file:///some/localpath/a_dataset') as reader:
       row_tensors = tf_tensors(reader)
       with tf.Session() as session:
           for _ in range(3):
               print(session.run(row_tensors))

The ``petastorm.reader.Reader`` class is the main entry point for user
code that accesses the data from an ML framework such as Tensorflow or Pytorch.
The reader has multiple features such as:

- Selective column readout
- Multiple parallelism strategies: thread, process, single-threaded (for debug)
- N-grams readout support
- Row filtering (row predicates)
- Shuffling
- Partitioning for multi-GPU training
- Local caching

Reading a dataset using Pytorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As illustrated in
`pytorch_example.py <https://github.com/uber/petastorm/blob/master/examples/mnist/pytorch_example.py>`_,
reading a petastorm dataset from pytorch
can be done via the adapter class ``petastorm.pytorch.DataLoader``,
which allows custom pytorch collating function and transforms to be supplied.

Be sure you have ``torch`` and ``torchvision`` installed:

.. code-block:: bash

    pip install torchvision

The minimalist example below assumes the definition of a ``Net`` class and
``train`` and ``test`` functions, included in ``pytorch_example``:

.. code-block:: python

    import torch
    from petastorm.pytorch import DataLoader

    torch.manual_seed(1)
    device = torch.device('cpu')
    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    def _transform_row(mnist_row):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return (transform(mnist_row['image']), mnist_row['digit'])

    with DataLoader(Reader('file:///localpath/mnist/train', num_epochs=10),
                    batch_size=64, transform=_transform_row) as train_loader:
        train(model, device, train_loader, 10, optimizer, 1)
    with DataLoader(Reader('file:///localpath/mnist/test', num_epochs=10),
                    batch_size=1000, transform=_transform_row) as test_loader:
        test(model, device, test_loader)

.. inclusion-marker-end-do-not-remove

Querying a dataset using Spark SQL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using the Parquet data format, which is natively supported by Spark, makes it possible to use a wide range of Spark
tools to analyze and manipulate the dataset. The example below shows how to read a Petastorm dataset
as a Spark RDD object:

.. code-block:: python

   # Create a dataframe object from a parquet file
   dataframe = spark.read.parquet(dataset_url)

   # Show a schema
   dataframe.printSchema()

   # Count all
   dataframe.count()

   # Show a single column
   dataframe.select('id').show()

SQL can be used to query a Petastorm dataset:

.. code-block:: python

   spark.sql(
      'SELECT count(id) '
      'from parquet.`file:///tmp/hello_world_dataset`').collect()

You can find a full code sample here: `pyspark_hello_world.py <https://github.com/uber/petastorm/blob/master/examples/hello_world/pyspark_hello_world.py>`_,

Troubleshooting
---------------

See the Troubleshooting_ page and please submit a ticket_ if you can't find an
answer.


Development
-----------

See the Development_ page for instructions on how to develop Petastorm and
run tests.


.. _Troubleshooting: docs/troubleshoot.rst
.. _ticket: https://github.com/uber/petastorm/issues/new
.. _Development: docs/development.rst
