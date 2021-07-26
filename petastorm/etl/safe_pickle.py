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

import io
import pickle

# List of packages which symbols are allowed to be loaded
safe_modules = {
    "petastorm",
    "collections",
    "numpy",
    "pyspark",
    "decimal",
    "builtins",
}


class RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        """Only allow safe classes from builtins"""
        package_name = module.split(".")[0]
        if package_name in safe_modules:
            return super().find_class(module, name)
        else:
            # Forbid everything else
            raise pickle.UnpicklingError(f"global {module, name} is forbidden")


def safe_loads(s):
    """Helper function analogous to pickle.loads()"""
    return RestrictedUnpickler(io.BytesIO(s)).load()
