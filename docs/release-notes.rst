.. inclusion-marker-start-do-not-remove

=============
Release notes
=============

Release 0.9.3 (unreleased)
==========================

Major features
--------------
- Retire support for Python 2.
- `PR 568 <https://github.com/uber/petastorm/pull/568>`_: Added additional kwargs for Spark Dataset Converter.


Release 0.9.2
=============
- `PR 556 <https://github.com/uber/petastorm/pull/556>`_: Remove driver param for hdfs.connect when using pyarrow 0.17 and above.


Release 0.9.1
=============

Bug fixes and additional features
---------------------------------
- `PR 554 <https://github.com/uber/petastorm/pull/554>`_: Fixed ``make_petastorm_dataset`` compatibility issue with ``WeightedSamplingReader``.
- `PR 535 <https://github.com/uber/petastorm/pull/535>`_: Stip trailing ``/`` from directory names when using ``GCSFSWrapper``'s ``walk``.


Release 0.9.0
=============

Thanks to our contributors: Weichen Xu, Liang Zhang, Felipe Petroski Such, Jennifer Blight and Greg Arutiunian!

Major features
--------------

- Apache PySpark 3.0 (Preview) support
- TensorFlow 2 support
- PR 540: petastorm.pytorch.BatchedDataLoader is a much faster version of petastorm.pytorch.DataLoader. petastorm.pytorch.BatchedDataLoader support only types that can be converted to pytorch tensors.
- Spark Dataset Converter: Spark converter API simplifies the data conversion from Spark to TensorFlow or PyTorch. The input Spark DataFrame is first materialized in the parquet format and then loaded as a tf.data.Dataset or torch.utils.data.DataLoader.

Bug fixes and additional features
---------------------------------
- `PR 513 <https://github.com/uber/petastorm/pull/513>`_: Preserve spark dataframe schema order in a namedtuples produced by `make_reader` and `make_batch_reader`. A backward compatiility flag is `here <https://github.com/uber/petastorm/blob/63c1faf9de3ebc56a386117bdd68deccd896fe25/petastorm/unischema.py#L37>`_.
- `PR 490 <https://github.com/uber/petastorm/pull/490>`_: TransformSpecs was not being applied if there was also a row predicate specified when using make_reader.
- `PR 495 <https://github.com/uber/petastorm/pull/495>`_: Removed spurious warning: “schema_fields behavior has changed. Now, regular expression pattern must match the entire field name...the following fields: “
- `PR 498 <https://github.com/uber/petastorm/pull/498>`_: Raise an explicit error if a non-scalar value is used as a value for a scalar field.
- `PR 529 <https://github.com/uber/petastorm/pull/529>`_: Using parquet.summary.metadata.level to control summary metadata file creation. This eliminates depreciation warning.
- `PR 533 <https://github.com/uber/petastorm/pull/533>`_:  make_petastorm_dataset (TF tf.data API) function supports ngrams.
- `PR 544 <https://github.com/uber/petastorm/pull/544>`_: Enable multiple iterations over the same petastorm.pytorch.DataLoader instance.

Python 2 support is deprecated and it will be removed in the next release.


Release 0.8.2
=============

New features and bug fixes
--------------------------
- pandas 1.0 compatibility issue resolved.

Release 0.8.1
=============

Thanks to our new contributor: Sergey Serebryakov for contributing Google Cloud Storage (GCS) filesystem implementation.

New features and bug fixes
--------------------------
- Bug fix: when ``make_batch_reader`` was used and Parquet files in a dataset have different number of row-groups. Symptoms:

  - ``pyarrow.lib.ArrowIOError: The file only has <X> row groups, requested metadata for row group: <Y>`` (where <X> and <Y> are integers) error message;
  - loss of some rows in a dataset.

  The issue was introduced in petastorm 0.7.7. `PR 477 <https://github.com/uber/petastorm/pull/477>`_.
- Google Cloud Storage (GCS) is now supported.
- Fixed a bug: ``AttributeError: 'ParseResult' object has no attribute 'hostname'``
- Fixed pyarrow 0.14.1 / PY2.7 compatibility issue: ``TypeError: __new__() got an unexpected keyword argument 'datestr'``
- Added ``image_codec`` property to ``CompressedImageCodec`` class.


Release 0.8.0
=============

Breaking changes
----------------
 - Regex specified in ``schema_fields`` argument of ``make_reader`` and ``make_batch_reader`` must match the
   entire field name.

   - Before this change: field named ``field_123`` would be loaded from a dataset
     by a reader ``make_reader(..., schema_fields=['123'])``.
   - After this change, user will need to specify a regex to match the full field name, for example:

     - ``make_reader(..., schema_fields=['field_123'])``; or
     - ``make_reader(..., schema_fields=['.*123'])``


New features and bug fixes
--------------------------
 - Fixed pyarrow 0.13 and 0.14 compatibility bugs.
 - Avoid using zero-memory-copy zmq feature of ``recv_multipart``. Otherwise, wild memory footprint swings
   in the main process are observed due to python GC behavior.

Release 0.7.7
=============

New features and bug fixes
--------------------------
 - Bug fix: use shape/type/codec information stored in the dataset even if a user passes a ``UnischemaField`` instance as
   ``make_reader``'s ``schema_field`` argument that may have contradicting shape/type/codec setup.
 - Resurrected ``WeightedSamplingReader`` class.
 - Make ``codec`` and ``nullable`` parameters of ``UnischemaField`` constructor optional (the codec can be automatically
   configured based on the ``numpy_dtype`` value).
 - Issue a warning if ``TransformSpec`` is configured with non-existing field names in ``remove_fields``.
 - Raise an error if trying to read from a ``Reader`` if reading after it was stopped (or goes out of context_manager
   scope).
 - Resolved pyarrow 0.15.0 compatibility issues.

Release 0.7.6
=============
Thanks to our new contributor: Michael Noronha and Praateek Mahajan!

New features and bug fixes
--------------------------
- Adding shuffling queue support to Petastorm's ``DataLoader``. Reordering of samples will occure when
  Petastorm's ``DataLoader`` is instantiated with ``shuffling_queue_capacity`` argument.
- Apply transform function after predicate was evaluated (with make_reader). It was reported in #391, that having the
  transform applied before predicate does not allow us to get rid of string fields that can not be forwarded
  to PyTorch. This PR changes the order of predicate/transform so that the scenario described in #391 works.
- Make sure ``in_pseudorandom_split`` works also with non-string fields.
- Improve support of reading from a partitioned dataset: interpret types of the partition-by scalars properly.
- Remove dependency on pyspark while reading using ``make_batch_reader``.
- Automatically delete columns when ``TransformSpec(..., removed_fields=..)`` is specified. Now user does not need
  to guarantee that the field is removed by the transform-spec function implementation.
- Read proper batches when using ``petastorm.pytorch.DataLoader`` with ``make_batch_reader``. Before this fix, batches
  were not formed properly.


Release 0.7.5
=============
New features and bug fixes
--------------------------
- Deadcode removal: ReaderV2 implementation is removed from the codebase.
- Avoid accidentally capturing FilesystemResolver when dataset metadata is generated
  (a call from ``materialize_dataset``)
- When possible, use spark name acquired from spark context as a username for hdfs connect call.
- Allow username to be passed to filesystem resolver and HdfsConnector/HAHdfsClient that they can provide to pyarrow hdfs connect call.


Release 0.7.4
=============
Thanks to our new contributor: Greg Arutiunian!

New features and bug fixes
--------------------------
- Support uint32 type when using `tf_tensor` and `make_petastorm_dataset`.
- Raise and error when trying to shard dataset and one of the shards end up being empty.
- ``IntersectIndexSelector`` enables to select row groups containing any of the values in all
  given selectors.
- ``UnionIndexSelector`` enable to select row groups containing any of the values in at least
  one selector.
- Regular expression can be used to select fields in an ``NGram`` specification.
- Fix for a crash in ``LocalDiskArrowTableCache`` occuring in some scenarios.


Release 0.7.3
=============
New features and bug fixes
--------------------------
- Adding missing ``future`` package as a wheel dependency


Release 0.7.2
=============


New features and bug fixes
--------------------------
- Add `Reader.reset()` method. Resets ``Reader`` state and allows to fetch more samples once the ``Reader`` finished reading all epochs,
  as specified by the ``num_epochs`` parameter. Once all samples were read from a reader, an attempt to fetch new sample (e.g. ``next(reader)`` would raise
  ``StopIterationError``. You can reset the reader to the original state and restart reading samples
  calling ``reset()``.
- pyarrow 0.13 compatibility issue fixed (``TypeError: an integer is required`` error was raised before the fix)
- ``make_batch_reader`` can now be used with Parquet stores that have ``name`` as a column name.
- Other ``make_batch_reader`` bug fixes.


Release 0.7.1
=============

Thanks to our contributors: Rémy Saissy!

New features and bug fixes
--------------------------
- Support opening parquet datasets with more than 255 columns using Python 3.0-3.6.
- Do not cast integers scalars to floats in columns containing nullable integer values.
- Include column name in an exception raised in case of decoding failure.



Release 0.7.0
=============

Thanks to our new contributer: Rémy Saissy!

Breaking changes
----------------
- ``transform`` argument of the ``petastorm.pytorch.DataLoader`` class was superseeded by `transform_spec` argument of
  ``make_reader``/``make_batch_reader`` functions.

New features and bug fixes
--------------------------
- `pyarrow==0.12.1` compatibility issues fixes.
- Ignore list-of-lists and list-of-structs columns (with a warning) when loading data from Apache Parquet store.
  This is useful when reading from an existing Parquet store that has these incompatible types.
- Added a mutex synchronizing readout from the results queue. Previously, race condition could have caused loss of
  some samples when fetching samples from ``Reader`` object from multiple threads.


Release 0.6.0
=============

Thanks to our new contributers: Kim Hammar and Joshua Goller!

Breaking changes
----------------
- ``petastorm.etl.dataset_metadata.materialize_dataset()`` should be passed a filesystem factory method
  instead of a pyarrow filesystem object. This change was made to fix a serialization bug that occurred during
  distributed reads (#280)

New features and bug fixes
--------------------------
- Added functionality for transform-on-worker thread/pool. The transform enables PyTorch users to run preprocessing
  code on worker processes/threads. It enables Tensorflow users to parallelize Python preprocessing code on
  a process pool, as part of the training/evaluation graph. Users now specify a ``transform_spec`` when calling
  ``make_reader()`` or ``make_batch_reader()``.
- Added ``hdfs_driver`` argument to the following functions: ``get_schema_from_dataset_url``, ``FilesystemResolver``,
  ``generate_petastorm_metadata``, ``build_rowgroup_index``, ``RowGroupLoader``, ``dataset_as_rdd`` and ``copy_dataset``
- the Docker container in ``/docker`` has been made into a workspace container aimed to support development on MacOS.
- New `hello_world` examples added for using non-Petastorm datasets.
- Allow for unicode strings to be passed as regex filters in Unischema when selecting which columns to read.
- Fixed a bug that caused all columns of a dataset to be read when ``schema_fields=NGram(...)`` was used.
- Fixed type of an argument passed to a predicate when the predicate is defined on a numeric partition field
- Support regular unicode strings as expressions as a value of make_reader's schema_fields argument.
- Emit a warning when opening a Petastorm-created dataset using make_batch_reader (``make_batch_reader`` currently
  does not support Petastorm specific types, such as tensors).

Release 0.5.1
=============

Breaking changes
----------------
None

New features and bug fixes
--------------------------
- ``make_batch_reader`` and ``make_reader`` now take an optional ``schema_fields`` argument. The argument may contain
  a list of field names or regular expression patterns that define a set of columns loaded from a parquet store.
- The following data types are now supported when opening a non-Petastorm Parquet store using ``make_batch_reader``:

  - ``DateType``
  - ``TimestampType``
  - ``ArrayType``


Release 0.5.0
=============

Breaking changes
----------------
- :func:`~petastorm.reader.make_reader` should be used to create new instance of a reader.
- It is still possible, but not recommended to use :class:`~petastorm.reader.Reader` in most cases. Its constructor arguments
  has changed:

 - ``training_partition`` and ``num_training_partitions`` were renamed into ``cur_shard`` and ``shard_count``.
 - ``shuffle`` and ``shuffle_options`` were replaced by ``shuffle_row_groups=True, shuffle_row_drop_partitions=1``
 - ``sequence`` argument was removed


New features and bug fixes
--------------------------
- It is possible to read non-Petastorm Parquet datasets (created externally to Petastorm). Currently most of the
  scalar types are supported.
- Support s3 as the protocol in a dataset url strings (e.g. 's3://...')
- PyTorch: support collating decimal scalars
- PyTorch: promote integer types that are not supported by PyTorch to the next larger integer types that is supported
  (e.g. int8 -> int16). Booleans are promoted to uint8.
- Support running ``petastorm-generate-metadata.py`` on datasets created by Hive.
- Fix incorrect dataset sharding when using Python 3.

Release 0.4.3
=============

New features and bug fixes
--------------------------
- A command line utility ``petastorm-copy-dataset.py``. Makes it easier to create subsets (columns/rows) of existing petastorm datasets.
- Add option to use custom pyarrow filesystem when materializing datasets.
- Limit memory usage correctly when using ``Reader`` with ``ProcessPool``.
- Added ``--pyarrow-serialize`` switch to ``petastorm-throughput.py`` benchmarking command line utility.
- Faster serialization (using ``pyarrow.serialize``) in ``ProcessPool`` implementation. Now decimal types are supported.
- More information in reader.diagnostics property.
- Check if a --unischema string passed to petastorm-generate-metadata is actually a ``Unischema`` instance.
- Fixed race condition in ``ProcessPool`` resulting in indefinite wait on ``ProcessPool`` shutdown.
- Force loading ``pyarrow`` before ``torch``. Helps to avoid a segfault (documented in ``docs/toubleshoot.rst``)
- Fixed mnist training examples.
- Make dependency on opencv optional in ``codecs.py``



Release 0.4.2
=============

New features and bug fixes
--------------------------
- Making ``decimal.Decimal`` to be decoded as ``decimal``, as in before 0.4.0.
- Adding a ``benchmark`` module with a ``petastorm-throughput.py`` command line utility.

Release 0.4.0, 0.4.1
====================

Breaking changes
----------------
- All ``decimal.Decimal`` fields are now decoded as strings
- ``PredicateBase`` moved from ``petastorm`` package to ``petastorm.predicates``
- ``RowGroupSelectorBase`` moved from ``petastorm`` package to ``petastorm.selectors``

New features and bug fixes
--------------------------
- Added ``WeightedSamplingReader``: aggregates multiple ``Reader`` output by sampling them with a specified probabilityWeightedSamplingReader `see documentation <https://petastorm.readthedocs.io/en/v0.4.0/reader.html#petastorm.weighted_sampling_reader.WeightedSamplingReader>`_.
- Add option for driver memory in regenerating metadata
- petastorm-generate-metadata command line tool renamed to petastorm-generate-metadata.py
- pytorch support (``petastorm.pytorch.DataLoader`` class)
- pytorch and tensorflow mnist model training
- Added ``CompressedNdarrayCodec`` codec
- Support passing pyarrow filesystem as ``Reader`` construction argument
- Speedup serialization (use ``pyarrow.serialize``) when ProcessPool is used.
- New, experimental, implementation of reader: ``ReaderV2``.
