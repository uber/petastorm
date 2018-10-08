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

###
# Adapted to petastorm dataset using original contents from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
###

from __future__ import division, print_function

import argparse
import os

import tensorflow as tf

from examples.mnist import DEFAULT_MNIST_DATA_PATH
from petastorm import make_reader
from petastorm.tf_utils import tf_tensors


def train_and_test(dataset_url, training_iterations, batch_size, evaluation_interval):
    """
    Train a model for training iterations with a batch size batch_size, printing accuracy every log_interval.
    :param dataset_url: The MNIST dataset url.
    :param training_iterations: The training iterations to train for.
    :param batch_size: The batch size for training.
    :param evaluation_interval: The interval used to print the accuracy.
    :return:
    """
    with make_reader(os.path.join(dataset_url, 'train'), num_epochs=None) as train_reader:
        with make_reader(os.path.join(dataset_url, 'test'), num_epochs=None) as test_reader:
            train_readout = tf_tensors(train_reader)
            train_image = tf.cast(tf.reshape(train_readout.image, [784]), tf.float32)
            train_label = train_readout.digit
            batch_image, batch_label = tf.train.batch(
                [train_image, train_label], batch_size=batch_size
            )

            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))
            y = tf.matmul(batch_image, W) + b

            # The raw formulation of cross-entropy,
            #
            #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
            #                                 reduction_indices=[1]))
            #
            # can be numerically unstable.
            #
            # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
            # outputs of 'y', and then average across the batch.
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=batch_label, logits=y)
            train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

            correct_prediction = tf.equal(tf.argmax(y, 1), batch_label)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            test_readout = tf_tensors(test_reader)
            test_image = tf.cast(tf.reshape(test_readout.image, [784]), tf.float32)
            test_label = test_readout.digit
            test_batch_image, test_batch_label = tf.train.batch(
                [test_image, test_label], batch_size=batch_size
            )

            # Train
            print('Training model for {0} training iterations with batch size {1} and evaluation interval {2}'.format(
                training_iterations, batch_size, evaluation_interval
            ))
            with tf.Session() as sess:
                sess.run([
                    tf.local_variables_initializer(),
                    tf.global_variables_initializer(),
                ])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                try:
                    for i in range(training_iterations):
                        if coord.should_stop():
                            break

                        sess.run(train_step)

                        if (i % evaluation_interval) == 0 or i == (training_iterations - 1):
                            feed_batch_image, feed_batch_label = sess.run([test_batch_image, test_batch_label])
                            print('After {0} training iterations, the accuracy of the model is: {1:.2f}'.format(
                                i,
                                sess.run(accuracy, feed_dict={
                                    batch_image: feed_batch_image, batch_label: feed_batch_label
                                })))
                finally:
                    coord.request_stop()
                    coord.join(threads)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Petastorm Tensorflow MNIST Example')
    default_dataset_url = 'file://{}'.format(DEFAULT_MNIST_DATA_PATH)
    parser.add_argument('--dataset-url', type=str,
                        default=default_dataset_url, metavar='S',
                        help='hdfs:// or file:/// URL to the MNIST petastorm dataset'
                             '(default: %s)' % default_dataset_url)
    parser.add_argument('--training-iterations', type=int, default=100, metavar='N',
                        help='number of training iterations to train (default: 100)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--evaluation-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before evaluating the model accuracy (default: 10)')
    args = parser.parse_args()

    train_and_test(
        dataset_url=args.dataset_url,
        training_iterations=args.training_iterations,
        batch_size=args.batch_size,
        evaluation_interval=args.evaluation_interval,
    )


if __name__ == '__main__':
    main()
