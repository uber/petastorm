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

from __future__ import division

import numpy as np


class WeightedSamplingReader(object):
    """Allows to combine outputs of two or more Reader objects, sampling them with a configurable probability.
    Complies to the same interfaces as :class:`~petastorm.reader.Reader`, hence
    :class:`~petastorm.weighted_sampling_reader.WeightedSamplingReader` can be used anywhere the
    :class:`~petastorm.reader.Reader` can be used."""

    def __init__(self, readers, probabilities):
        """Creates an instance WeightedSamplingReader.

        The constructor gets a list of readers and probabilities as its parameters. The lists must be the same length.
        :class:`~petastorm.weighted_sampling_reader.WeightedSamplingReader` implements an iterator interface. Each time
        a new element is requested, one of the readers is selected, weighted by the matching probability. An element
        produced by the selected reader is returned.

        The iterator raises StopIteration exception once one of the embedded readers has no more data left.

        The following example shows how a :class:`~petastorm.weighted_sampling_reader.WeightedSamplingReader` can be
        instantiated with two readers which are sampled with 10% and 90% probabilities respectively.

        >>> from petastorm.weighted_sampling_reader import WeightedSamplingReader
        >>> from petastorm.reader import Reader
        >>>
        >>> with WeightedSamplingReader([Reader('file:///dataset1'), Reader('file:///dataset1')], [0.1, 0.9]) as reader:
        >>>     new_sample = next(reader)


        :param readers: A list of readers. The length of the list must be the same as the length of the
         ``probabilities`` list.
        :param probabilities: A list of probabilities. The length of the list must be the same as the length
          of ``readers`` argument. If the sum of all probability values is not 1.0, it will be automatically
          normalized.

        """
        if len(readers) != len(probabilities):
            raise ValueError('readers and probabilities are expected to be lists of the same length')

        self._readers = readers

        # Normalize probabilities
        self._cum_prob = np.cumsum(np.asarray(probabilities, dtype=np.float) / np.sum(probabilities))

    def __len__(self):
        return sum(len(reader) for reader in self._readers)

    def __iter__(self):
        return self

    def __next__(self):
        r = np.random.random()
        reader_index = np.where(r < self._cum_prob)[0][0]
        return next(self._readers[reader_index])

    def next(self):
        return self.__next__()

    # Functions needed to treat reader as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for reader in self._readers:
            reader.stop()

        for reader in self._readers:
            reader.join()
