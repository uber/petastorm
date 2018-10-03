from examples.mnist import tf_example as tf_example
from examples.mnist.generate_petastorm_mnist import mnist_data_to_petastorm_dataset


def test_full_tf_example(large_mock_mnist_data, tmpdir):
    # First, generate mock dataset
    dataset_url = 'file://{}'.format(tmpdir)
    mnist_data_to_petastorm_dataset(tmpdir, dataset_url, mnist_data=large_mock_mnist_data,
                                    spark_master='local[1]', parquet_files_count=1)

    # Tensorflow train and test
    tf_example.train_and_test(
        dataset_url=dataset_url,
        training_iterations=10,
        batch_size=10,
        evaluation_interval=10,
    )
