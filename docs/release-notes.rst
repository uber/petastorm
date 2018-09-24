.. inclusion-marker-start-do-not-remove

=============
Release notes
=============

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
