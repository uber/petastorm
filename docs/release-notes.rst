.. inclusion-marker-start-do-not-remove

=============
Release notes
=============

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
