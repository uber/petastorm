#  Copyright (c) 2020 Databricks, Inc.
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

###
# Adapted to spark_dataset_converter using original contents from
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb
# This example runs with PySpark > 3.0.0
###
import logging

from pyspark.sql import SparkSession

from examples.spark_dataset_converter.utils import get_mnist_dir
from petastorm.spark import SparkDatasetConverter, make_spark_converter

try:
    from pyspark.sql.functions import col
except ImportError:
    raise ImportError("This script runs with PySpark>=3.0.0")


def get_compiled_model(lr=0.001):
    from tensorflow import keras

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10),
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def train(dataset, steps=1000, lr=0.001):
    model = get_compiled_model(lr=lr)
    model.fit(dataset, steps_per_epoch=steps)
    return model


def run(data_dir):
    # Get SparkSession
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("petastorm.spark tensorflow_example") \
        .getOrCreate()

    # Load and preprocess data using Spark
    df = spark.read.format("libsvm") \
        .option("numFeatures", "784") \
        .load(data_dir) \
        .select(col("features"), col("label").cast("long").alias("label"))

    # Randomly split data into train and test dataset
    df_train, df_test = df.randomSplit([0.9, 0.1], seed=12345)

    # Set a cache directory for intermediate data.
    # The path should be accessible by both Spark workers and driver.
    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
                   "file:///tmp/petastorm/cache/tf-example")

    converter_train = make_spark_converter(df_train)
    converter_test = make_spark_converter(df_test)

    def train_and_evaluate(_=None):
        import tensorflow.compat.v1 as tf  # pylint: disable=import-error

        with converter_train.make_tf_dataset() as dataset:
            dataset = dataset.map(lambda x: (tf.reshape(x.features, [-1, 28, 28]), x.label))
            model = train(dataset)

        with converter_test.make_tf_dataset(num_epochs=1) as dataset:
            dataset = dataset.map(lambda x: (tf.reshape(x.features, [-1, 28, 28]), x.label))
            hist = model.evaluate(dataset)

        return hist[1]

    # Train and evaluate the model on the local machine
    accuracy = train_and_evaluate()
    logging.info("Train and evaluate the model on the local machine.")
    logging.info("Accuracy: %.6f", accuracy)

    # Train and evaluate the model on a spark worker
    accuracy = spark.sparkContext.parallelize(range(1)).map(train_and_evaluate).collect()[0]
    logging.info("Train and evaluate the model remotely on a spark worker, "
                 "which can be used for distributed hyperparameter tuning.")
    logging.info("Accuracy: %.6f", accuracy)

    # Cleanup
    converter_train.delete()
    converter_test.delete()
    spark.stop()


def main():
    mnist_dir = get_mnist_dir()
    run(data_dir=mnist_dir)


if __name__ == '__main__':
    main()
