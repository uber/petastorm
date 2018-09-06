.. inclusion-marker-start-do-not-remove

============================
Migrating to petastorm 0.3.0
============================

Petastorm 0.3.0 has some breaking changes from previous versions. These include:

- Change of :class:`Reader` argument order, making ``dataset_url`` required and first
- Deprecation of :class:`Sequence` in favor of :class:`NGram`
- Using :func:`materialize_dataset` when writing out a dataset as opposed to the deprecated/removed :func:`add_dataset_metadata` function

To change :class:`Reader` argument order
========================================

This is the simplest breaking change in 0.3.0. Simply change the argument order from:

.. code-block:: python

    from petastorm.reader import Reader
    reader = Reader(schema_fields, dataset_url)

To:

.. code-block:: python

    from petastorm.reader import Reader
    reader = Reader(dataset_url, schema_fields=schema_fields)

To change from :class:`Sequence` to :class:`NGram`
==================================================

Using :class:`Sequence`

.. code-block:: python

    from petastorm.reader import Reader
    from petastorm.sequence import Sequence

    sequence = Sequence(length=5, delta_treshold=10, timestamp_field='timestamp')
    schema_fields = MySchema.fields
    reader = Reader(dataset_url=dataset_url, schema_fields=schema_fields, sequence=sequence)
    ...

Equivalent results using :class:`NGram`

.. code-block:: python

    from petastorm.reader import Reader
    from petastorm.ngram import NGram

    ngram_fields = NGram(
      fields={step: MySchema.fields for step in range(5)},
      delta_treshold=10,
      timestamp_field=MySchema.timestamp
    )
    reader = Reader(dataset_url, schema_fields=ngram_fields)
    ...

Note that if you do not need all fields in each step of the NGram, it will be adventageous to define the NGram fields
with what you actually require. However this code snippet will provide the exact same result as previous petastorm versions
if you are looking for a drop in replacement.

Using :func:`materialize_dataset`
=================================

In Petastorm 0.3.0 we have deprecated and removed :func:`add_dataset_metadata` which was previously used to add the 
schema data and other necessary metadata required to read petastorm datasets. We have instead created a context manager
which should be used around your spark write to add this necessary data. The context manager gives us more control over
parquet options to set when writing the data, which is necessary as we move to using parquet summary files for getting
row group information.

Using :func:`add_dataset_metadata`

.. code-block:: python

    from petastorm.etl.dataset_metadata import add_dataset_metadata

    spark = SparkSession.Builder.getOrCreate()
    sc = spark.sparkContext
    dataframe = ...
    dataframe.write.parquet(dataset_url)
    add_dataset_metadata(dataset_url, sc, MySchema)


Using :func:`materialize_dataset`

.. code-block:: python

    from petastorm.etl.dataset_metadata import materialize_dataset

    spark = SparkSession.Builder.getOrCreate()
    sc = spark.sparkContext
    with materialize_dataset(spark, dataset_url, MySchema):
      dataframe = ...
      dataframe.write.parquet(dataset_url)

Updating metadata of existing datasets
======================================

We changed how dataset metadata is stored, which will likely result in a warning message of:
``You are using a deprecated metadata version. Please run petastorm-generate-metadata.py on spark to update.``

This is just recommending you to regenerate the metadata for any existing petastorm datasets to use the new structure.
Luckily while in the past it took several minutes to regenerate metadata for a dataset, it now should only take a few
seconds. There is a script ``petastorm-generate-metadata.py`` included in petastorm 0.3.0 which makes this easy.
Simply run:

.. code-block:: bash

    petastorm-generate-metadata.py --dataset_url hdfs://namenode-host:port/path/to/dataset

Which will regenerate the dataset metadata, reusing the schema information already there. By default this
will run on your local pyspark which should be fine given that the job does not utilize any executors
(only a driver). Note that pyspark running locally seems to have trouble picking up hadoop configuration files
(e.g. ``core-site.xml`` or ``hdfs-site.xml``) so it might be necessary to supply the namenode hostname directly instead
of using a nameservice or relying on defaults set in those files.

If you are running this on your local machine, also note that it will run under your user. If you are running this on
a location in remote storage (e.g. HDFS) you will need write permission to both the dataset directory and the
``_common_metadata``/``_metadata`` files (if they exist). The easiest way to accomplish this is to make the directory writeable
to your user either by changing file permissions, changing file ownership, etc. The easiest way to guarantee success would
be to ``chmod -R`` the directory to ``777`` however do this at your own risk. Another option would be to run
``petastorm-generate-metadata.py`` from within spark to have it share the same runtime environment permissions as the
original dataset run.


Updating image codec parameter name in :class:`CompressedImageCodec` constructor
================================================================================

We changed the ``format`` argument name to ``image_codec`` in :class:`CompressedImageCodec`'s constructor since ``format`` is a 
Python keyword. Example:

Old argument name:

.. code-block:: python

    CompressedImageCodec(format='jpeg')

New argument name:

.. code-block:: python

    CompressedImageCodec(image_codec='jpeg')
