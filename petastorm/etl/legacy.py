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

import logging

from six.moves import cPickle as pickle

import io

# List of packages which symbols are allowed to be loaded
safe_modules = {
    "petastorm",
    "collections",
    "numpy",
    "pyspark",
    "decimal",
    "builtins",
    "copy_reg",
    "__builtin__",
}


class RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        """Only allow safe classes from builtins"""
        package_name = module.split(".")[0]
        if package_name in safe_modules:
            return super().find_class(module, name)
        else:
            # Forbid everything else
            raise pickle.UnpicklingError("global '%s.%s' is forbidden" % (module, name))


def restricted_loads(s):
    """Helper function analogous to pickle.loads()"""
    return RestrictedUnpickler(io.BytesIO(s)).load()


logger = logging.getLogger(__name__)


def depickle_legacy_package_name_compatible(pickled_string):
    """Backward compatible way of depickling old pickled strings.

    Previously petastorm package was named differently. In order to be able to load older datasets, we modify
    module names in the pickled stream with the new ones.

    :param pickled_string: A pickled string to be passed to pickle.loads
    :return:
    """
    LEGACY_PACKAGE_NAMES = ['av.experimental.deepdrive.dataset_toolkit', 'av.ml.dataset_toolkit']
    LEGACY_MODULES = ['codecs', 'unischema', 'sequence']

    for legacy_package_name in LEGACY_PACKAGE_NAMES:
        for legacy_module in LEGACY_MODULES:
            # Substitute module names directly in the pickled stream. Encode as 'ascii' to make sure no non-ascii
            # character made its way into package/module name
            legacy_package_entry = '\n(c{}.{}\n'.format(legacy_package_name, legacy_module).encode('ascii')
            new_module_name = '\n(cpetastorm.{}\n'.format(legacy_module).encode('ascii')
            modified_pickled_string = pickled_string.replace(legacy_package_entry, new_module_name)
            if modified_pickled_string != pickled_string:
                logger.warning('Depickling "%s.%s" which has moved to "petastorm.%s". '
                               'Regenerate metadata.', legacy_package_name, legacy_module, legacy_module)

            pickled_string = modified_pickled_string

    return restricted_loads(pickled_string)
