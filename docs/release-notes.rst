.. inclusion-marker-start-do-not-remove

=============
Release notes
=============

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
