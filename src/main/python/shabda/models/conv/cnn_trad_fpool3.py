#Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py#L165

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import signal
from tqdm import tqdm
from sarvam.helpers.print_helper import *
from speech_recognition.dataset.feature_types import MFCCFeature
from speech_recognition.sr_config.sr_config import *
from tensorflow.contrib.learn import ModeKeys

class CNNTradFPoolConfig:
    def __init__(self):
        self._model_dir = "experiments/CNNTradFPool/"

        self._seed = 2018
        self._batch_size = 32
        self._keep_prob = 0.5
        self._learning_rate = 1e-3
        self._clip_gradients = 15.0
        self._use_batch_norm = True
        self._num_classes = len(POSSIBLE_COMMANDS) + 2

    @staticmethod
    def user_config():
        return CNNTradFPoolConfig()


class CNNTradFPool(tf.estimator.Estimator):
    def __init__(self,
                 sr_config, run_config):
        super(CNNTradFPool, self).__init__(
            model_fn=self._model_fn,
            model_dir=sr_config._model_dir,
            config=run_config)

        self.sr_config = sr_config
        self._feature_type = MFCCFeature

        audio_sampling_settings = self._feature_type.audio_sampling_settings

        self._dct_coefficient_count = audio_sampling_settings["dct_coefficient_count"]
        self._spectrogram_length = audio_sampling_settings["spectrogram_length"]

        self._label_count = audio_sampling_settings['label_count']

    def _model_fn(self, features, labels, mode, params):

        control_dependencies = []
        checks = tf.add_check_numerics_ops()
        control_dependencies = [checks]

        input_frequency_size = self._dct_coefficient_count
        input_time_size = self._spectrogram_length

        first_filter_width = 8
        first_filter_height = 20
        first_filter_count = 64

        second_filter_width = 4
        second_filter_height = 10
        second_filter_count = 64

        fingerprint_input = features[self._feature_type.FEATURE_1]
        labels = tf.cast(labels, tf.int64)

        tf.logging.info("=====> fingerprint_input {}".format(fingerprint_input))
        tf.logging.info("=====> labels {}".format(labels))

        fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size, input_frequency_size, 1])

        tf.logging.info("=====> fingerprint_4d {}".format(fingerprint_4d))


        first_weights = tf.Variable(tf.truncated_normal([first_filter_height, first_filter_width, 1, first_filter_count],
                                                         stddev=0.01))
        first_bias = tf.Variable(tf.zeros([first_filter_count]))

        first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1], 'SAME') + first_bias

        tf.logging.info("=====> first_conv {}".format(first_conv))

        first_relu = tf.nn.relu(first_conv)

        if mode != ModeKeys.INFER:
            first_dropout = tf.nn.dropout(first_relu, self.sr_config._keep_prob)
        else:
            first_dropout = first_relu

        max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


        second_weights = tf.Variable(
            tf.truncated_normal(
                [
                    second_filter_height, second_filter_width, first_filter_count,
                    second_filter_count
                ],
                stddev=0.01))
        second_bias = tf.Variable(tf.zeros([second_filter_count]))
        second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1], 'SAME') + second_bias
        second_relu = tf.nn.relu(second_conv)

        tf.logging.info("=====> second_conv {}".format(second_conv))

        if mode != ModeKeys.INFER:
            second_dropout = tf.nn.dropout(second_relu, self.sr_config._keep_prob)
        else:
            second_dropout = second_relu

        second_conv_shape = second_dropout.get_shape()
        second_conv_output_width = second_conv_shape[2]
        second_conv_output_height = second_conv_shape[1]
        second_conv_element_count = int(
            second_conv_output_width * second_conv_output_height *
            second_filter_count)
        flattened_second_conv = tf.reshape(second_dropout,
                                           [-1, second_conv_element_count])

        final_fc_weights = tf.Variable(
            tf.truncated_normal(
                [second_conv_element_count, self.sr_config._num_classes], stddev=0.01))

        final_fc_bias = tf.Variable(tf.zeros([self.sr_config._num_classes]))
        logits = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias

        tf.logging.info("=====> logits {}".format(logits))

        if mode != ModeKeys.INFER:
            with tf.name_scope('cross_entropy'):
                losses = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            tf.logging.info("=====> losses {}".format(losses))
        classes = tf.argmax(logits, 1)
        predictions = {"classes": classes}

        tf.logging.info("=====> classes {}".format(classes))


        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}

        if mode != ModeKeys.INFER:
            with tf.name_scope('train-optimization'):#, tf.control_dependencies(control_dependencies):
                global_step = tf.train.get_global_step()
                learning_rate = self.sr_config._learning_rate
                train_op = tf.contrib.layers.optimize_loss(
                    loss=losses,
                    global_step=global_step,
                    optimizer=tf.train.GradientDescentOptimizer,
                    learning_rate=learning_rate)

                loss = losses

            correct_prediction = tf.equal(predictions["classes"], labels)
            confusion_matrix = tf.confusion_matrix(
                labels, predictions["classes"], num_classes=self.sr_config._num_classes)

            eval_metric_ops = {
                'Accuracy': tf.metrics.accuracy(
                    labels=tf.cast(labels, tf.int32),
                    predictions=predictions["classes"],
                    name='accuracy'),
                'Precision': tf.metrics.precision(
                    labels=tf.cast(labels, tf.int32),
                    predictions=predictions["classes"],
                    name='Precision'),
                'Recall': tf.metrics.recall(
                    labels=tf.cast(labels, tf.int32),
                    predictions=predictions["classes"],
                    name='Recall')
            }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            # training_hooks=self.hooks
        )