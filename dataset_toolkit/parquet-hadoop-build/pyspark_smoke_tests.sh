#!/bin/bash

PYTHON_TEST_LIST=$(cat <<END
source/datascience/tests/python/features/test_schema.py
source/experimental/deepdrive/python/dataset_toolkit/integration_tests/integration_extract_and_read_channels_and_targets.py
source/experimental/deepdrive/python/dataset_toolkit/tests/test_end_to_end.py
source/experimental/deepdrive/python/dataset_toolkit/tests/test_sequence_end_to_end.py
source/experimental/deepdrive/python/dataset_toolkit/tests/test_run_in_subprocess.py
source/experimental/deepdrive/python/dataset_toolkit/tests/test_unischema.py
source/experimental/deepdrive/python/dataset_toolkit/tests/test_tf_utils.py
source/perception/lidar_image_cnn_detector/tests/test_lidar_cnn_trainer.py
source/mlp/map_automation/tests/python/utils/test_spark_utils.py
source/spark/tests/pyspark_local_test.py
END
)

for pytest in ${PYTHON_TEST_LIST} ; do
    echo "==== Test: ${pytest}"
    ( ./bin/python ${pytest} && echo '==== PASSED! ====') || ( echo '====xxxx TEST FAILED!!! xxxx====' && exit -1 )
done
