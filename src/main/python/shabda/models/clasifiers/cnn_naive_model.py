# Copyright 2018 The Shabda Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Naive CNN Audio classifer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys
from overrides import overrides
from shabda.helpers.print_helper import *
from shabda.helpers import utils
from shabda.hyperparams.hyperparams import HParams
from shabda.models.clasifiers.classifer_base import ClassifierBase


class CustomDNN(ClassifierBase):
    def __init__(self, hparams=None):
        super(CustomDNN, self).__init__(hparams=hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        hparams = {
            "name": "cnn_naive",
            "out_dim": -1,
            "learning_rate": 0.001
        }
        return hparams

    @overrides
    def _build_layers(self, features, mode):
        tf.logging.debug('features -----> {}'.format(features))  # shape=(?, 64, 256, 1)

        float_audio_features = tf.cast(features, tf.float32, name="float_audio_features")
        audio_features = tf.identity(float_audio_features, name="audio_features")
        features_reshaped = tf.reshape(audio_features, shape=(-1, 64, 256, 1), name="features_reshaped")

        # features_reshaped = features
        tf.logging.debug('features_reshaped -----> {}'.format(features_reshaped))  # shape=(?, 64, 256, 1)

        with tf.name_scope('conv_layer'):
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=features_reshaped,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu,
                name="conv1")

            tf.logging.info('conv1: ------> {}'.format(conv1))  # (?, 64, 256, 32)

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=4)  # (?, 16, 64, 32)
            tf.logging.info('pool1: ------> {}'.format(pool1))

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu,
                name="conv2")
            tf.logging.info('conv2: ------> {}'.format(conv2))  # (?, 16, 64, 64)

            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=4)
            tf.logging.info('pool2: ------> {}'.format(pool2))  # (?, 4, 16, 64)

            # Dense Layer
            pool_flat = tf.reshape(pool2, [-1, 4 * 16 * 64], name="pool_flat_reshape")
            tf.logging.info('pool3_flat: ------> {}'.format(pool_flat))  # 60480

            dense = tf.layers.dense(inputs=pool_flat, units=512, activation=tf.nn.relu)

            # if mode != ModeKeys.INFER:
            #     dense = tf.layers.dropout(
            #         inputs=dense, rate=0.9, training=mode == tf.estimator.ModeKeys.TRAIN)
            #
            #     tf.logging.info('dense_layer: ------> {}'.format(dense))

        with  tf.name_scope("logits-layer"):
            # [?, self.NUM_CLASSES]

            logits = tf.layers.dense(inputs=dense,
                                     units=self._out_dim,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))

            tf.logging.info('logits: ------> {}'.format(logits))

            return logits

    #
    # def _model_fn(self, features, labels, mode):
    #     """Model function used in the estimator.
    #
    #     Args:
    #         features : Tensor(shape=[?], dtype=string) Input features to the model.
    #         labels : Tensor(shape=[?, n], dtype=Float) Input labels.
    #         mode (ModeKeys): Specifies if training, evaluation or prediction.
    #         params (HParams): hyperparameters.
    #
    #     Returns:
    #         (EstimatorSpec): Model to be run by Estimator.
    #     """
    #     print_debug(features)
    #
    #     audio_mfcc = features["features"]
    #
    #     print_debug(audio_mfcc)
    #     print_debug(features["labels"])
    #
    #     float_audio_features = tf.cast(audio_mfcc, tf.float32, name="float_audio_features")
    #     audio_features = tf.identity(float_audio_features, name="audio_features")
    #     features_reshaped = tf.reshape(audio_features, shape=(-1, 64, 256, 1), name="features_reshaped")
    #
    #
    #     if mode != ModeKeys.INFER:
    #         labels = features["labels"]
    #         labels = tf.identity(labels, name="features")
    #
    #     tf.logging.debug('features -----> {}'.format(audio_features)) #shape=(?, 64, 256, 1)
    #     tf.logging.debug('labels -----> {}'.format(labels))
    #
    #
    #     with  tf.name_scope('conv_layer'):
    #
    #         # Convolutional Layer #1
    #         conv1 = tf.layers.conv2d(
    #             inputs=features_reshaped,
    #             filters=32,
    #             kernel_size=[5, 5],
    #             padding="same",
    #             activation=tf.nn.relu,
    #             name="conv1")
    #
    #         tf.logging.info('conv1: ------> {}'.format(conv1))  # (?, 64, 256, 32)
    #
    #         # Pooling Layer #1
    #         pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=4)  # (?, 16, 64, 32)
    #         tf.logging.info('pool1: ------> {}'.format(pool1))
    #
    #         # Convolutional Layer #2 and Pooling Layer #2
    #         conv2 = tf.layers.conv2d(
    #             inputs=pool1,
    #             filters=64,
    #             kernel_size=[3, 3],
    #             padding="same",
    #             activation=tf.nn.relu,
    #             name="conv2")
    #         tf.logging.info('conv2: ------> {}'.format(conv2))  # (?, 16, 64, 64)
    #
    #         pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=4)
    #         tf.logging.info('pool2: ------> {}'.format(pool2))  # (?, 4, 16, 64)
    #
    #         # Dense Layer
    #         pool_flat = tf.reshape(pool2, [-1, 4 * 16 * 64], name="pool_flat_reshape")
    #         tf.logging.info('pool3_flat: ------> {}'.format(pool_flat)) #60480
    #
    #         dense = tf.layers.dense(inputs=pool_flat, units=512, activation=tf.nn.relu)
    #
    #         # if mode != ModeKeys.INFER:
    #         #     dense = tf.layers.dropout(
    #         #         inputs=dense, rate=0.9, training=mode == tf.estimator.ModeKeys.TRAIN)
    #         #
    #         #     tf.logging.info('dense_layer: ------> {}'.format(dense))
    #
    #     with  tf.name_scope("logits-layer"):
    #         # [?, self.NUM_CLASSES]
    #
    #         logits = tf.layers.dense(inputs=dense,
    #                                  units=self.labels_dim,
    #                                  kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))
    #
    #         tf.logging.info('logits: ------> {}'.format(logits))
    #
    #     with  tf.name_scope("output-layer"):
    #         # [?,1]
    #         predicted_class = tf.argmax(logits, axis=1, name="class_output")
    #         tf.logging.info('predicted_class: ------> {}'.format(predicted_class))
    #
    #         predicted_probabilities = tf.nn.softmax(logits, name="softmax_output")
    #         tf.logging.info('predicted_probabilities: ------> {}'.format(predicted_probabilities))
    #
    #     predictions = {
    #         "classes": predicted_class,
    #         "probabilities": predicted_probabilities
    #     }
    #
    #     # logging
    #     # self.log_tensors("output_probabilities", "output-layer/softmax_output")
    #     # tf.summary.histogram(encoding.name, encoding)
    #     tf.summary.histogram(predicted_probabilities.name, predicted_probabilities)
    #
    #     # Loss, training and eval operations are not needed during inference.
    #     loss = None
    #     train_op = None
    #     eval_metric_ops = {}
    #
    #     if mode != ModeKeys.INFER:
    #         tf.logging.info('labels: ------> {}'.format(labels))
    #         tf.logging.info('predictions["classes"]: ------> {}'.format(predictions["classes"]))
    #
    #         loss = tf.losses.softmax_cross_entropy(
    #             onehot_labels=labels,
    #             logits=logits,
    #             weights=0.80,
    #             scope='actual_loss')
    #
    #         loss = tf.reduce_mean(loss, name="reduced_mean")
    #
    #         train_op = tf.contrib.layers.optimize_loss(
    #             loss=loss,
    #             global_step=tf.train.get_global_step(),
    #             optimizer=tf.train.AdamOptimizer,
    #             learning_rate=0.001)
    #
    #         label_argmax = tf.argmax(labels, 1, name='label_argmax')
    #
    #         eval_metric_ops = {
    #             'Accuracy': tf.metrics.accuracy(
    #                 labels=label_argmax,
    #                 predictions=predictions["classes"],
    #                 name='accuracy'),
    #             'Precision': tf.metrics.precision(
    #                 labels=label_argmax,
    #                 predictions=predictions["classes"],
    #                 name='Precision'),
    #             'Recall': tf.metrics.recall(
    #                 labels=label_argmax,
    #                 predictions=predictions["classes"],
    #                 name='Recall')
    #         }
    #         tf.summary.scalar(loss.name, loss)
    #         # validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    #         #     test_set.audio_utils,
    #         #     test_set.target,
    #         #     every_n_steps=50,
    #         #     metrics=validation_metrics,
    #         #     early_stopping_metric="loss",
    #         #     early_stopping_metric_minimize=True,
    #         #     early_stopping_rounds=200)
    #
    #     return tf.estimator.EstimatorSpec(
    #         mode=mode,
    #         predictions=predictions,
    #         loss=loss,
    #         train_op=train_op,
    #         eval_metric_ops=eval_metric_ops
    #     )
