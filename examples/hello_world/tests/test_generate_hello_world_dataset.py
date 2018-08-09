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
import os
import shutil
import tempfile
import unittest

from examples.hello_world.hello_world_dataset import generate_hello_world_dataset


class TestGenerateHelloWorldDataset(unittest.TestCase):

    def test_generate(self):
        temp_dir = tempfile.mkdtemp()
        try:
            generate_hello_world_dataset('file://' + temp_dir)
            self.assertTrue('_SUCCESS' in os.listdir(temp_dir))
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
