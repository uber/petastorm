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

import pytest

from petastorm.spark import make_spark_converter
from petastorm.tests.test_tf_utils import _IS_TF_VERSION_1


@pytest.mark.skipif(_IS_TF_VERSION_1, reason="Only test autograph transform on tensorflow>=2")
def test_tf_autograph(spark_test_ctx, caplog):
    caplog.clear()
    df1 = spark_test_ctx.spark.range(100)
    converter1 = make_spark_converter(df1)
    results = []
    with converter1.make_tf_dataset(num_epochs=1) as dataset:
        for batch in dataset:
            results.append(batch)
    assert "AutoGraph could not transform" not in " ".join(caplog.messages)
