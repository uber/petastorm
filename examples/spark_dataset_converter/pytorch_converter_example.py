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
# https://github.com/pytorch/examples/blob/master/mnist/main.py
# This example runs with PySpark > 3.0.0
###
from __future__ import division

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pyspark.sql import SparkSession
from torch.autograd import Variable

from examples.spark_dataset_converter.utils import get_mnist_dir
from petastorm.spark import SparkDatasetConverter, make_spark_converter

try:
    from pyspark.sql.functions import col
except ImportError:
    raise ImportError("This script runs with PySpark>=3.0.0")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = x.view((-1, 1, 28, 28))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(data_loader, steps=100, lr=0.0005, momentum=0.5):
    model = Net()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_hist = []
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx > steps:
            break
        data, target = Variable(batch['features']), Variable(batch['label'])
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            logging.info('[%d/%d]\tLoss: %.6f', batch_idx, steps, loss.data.item())
            loss_hist.append(loss.data.item())
    return model


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    test_len = 0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch['features'], batch['label']
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_len += data.shape[0]

    test_loss /= test_len
    accuracy = correct / test_len

    logging.info('Test set: Average loss: %.4f, Accuracy: %d/%d (%.0f%%)',
                 test_loss, correct, test_len, 100. * accuracy)
    return accuracy


def run(data_dir):
    # Get SparkSession
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("petastorm.spark pytorch_example") \
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
                   "file:///tmp/petastorm/cache/torch-example")

    converter_train = make_spark_converter(df_train)
    converter_test = make_spark_converter(df_test)

    def train_and_evaluate(_=None):
        with converter_train.make_torch_dataloader() as loader:
            model = train(loader)

        with converter_test.make_torch_dataloader(num_epochs=1) as loader:
            accuracy = test(model, loader)
        return accuracy

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
