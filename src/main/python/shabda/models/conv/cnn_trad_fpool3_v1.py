# #Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py#L165
#
# import numpy as np
# import tensorflow as tf
# from tensorflow.contrib import layers
# from tensorflow.contrib import signal
# from tqdm import tqdm
# from shabda.helpers.print_helper import *
# from tensorflow.contrib.learn import ModeKeys
# from shabda.helpers.audio import prepare_audio_sampling_settings
#
# class CNNTradFPoolConfigV1:
#     def __init__(self):
#         self._model_dir = "models/CNNTradFPoolV1/"
#
#         self._seed = 2018
#         self._batch_size = 32
#         self._keep_prob = 0.5
#         self._learning_rate = 1e-3
#         self._clip_gradients = 15.0
#         self._use_batch_norm = True
#         self._num_classes = len(POSSIBLE_COMMANDS) + 2
#
#         audio_sampling_settings = prepare_audio_sampling_settings(label_count=10,
#                                                                   sample_rate=SAMPLE_RATE,
#                                                                   clip_duration_ms=CLIP_DURATION_MS,
#                                                                   window_size_ms=WINDOW_SIZE_MS,
#                                                                   window_stride_ms=WINDOW_STRIDE_MS,
#                                                                   dct_coefficient_count=DCT_COEFFICIENT_COUNT)
#
#         self._dct_coefficient_count = audio_sampling_settings["dct_coefficient_count"]
#         self._spectrogram_length = audio_sampling_settings["spectrogram_length"]
#
#         self._label_count = audio_sampling_settings['label_count']
#
#
# class CNNTradFPoolV1(tf.estimator.Estimator):
#     def __init__(self,
#                  sr_config, run_config):
#         super(CNNTradFPoolV1, self).__init__(
#             model_fn=self._model_fn,
#             model_dir=sr_config._model_dir,
#             config=run_config)
#
#         self.sr_config = sr_config
#
#     def _model_fn(self, features, labels, mode, params):
#
#         control_dependencies = []
#         checks = tf.add_check_numerics_ops()
#         control_dependencies = [checks]
#
#         input_frequency_size = self.sr_config._dct_coefficient_count
#         input_time_size = self.sr_config._spectrogram_length
#
#         first_filter_width = 8
#         first_filter_height = 20
#         first_filter_count = 64
#
#         second_filter_width = 4
#         second_filter_height = 10
#         second_filter_count = 64
#
#         fingerprint_input = features["wav"]
#         labels = tf.cast(labels, tf.int64)
#
#         tf.logging.info("=====> fingerprint_input {}".format(fingerprint_input))
#         tf.logging.info("=====> labels {}".format(labels))
#
#         fingerprint_4d = tf.reshape(fingerprint_input,
#                                     [-1, input_time_size, input_frequency_size, 1]) #[-1, 98, 40, 1]
#
#         tf.logging.info("=====> fingerprint_4d {}".format(fingerprint_4d))
#
#         with  tf.name_scope('conv_layer'):
#             # Convolutional Layer #1
#             conv1 = tf.layers.conv2d(
#                 inputs=fingerprint_4d,
#                 filters=32,
#                 kernel_size=[5, 5],
#                 padding="same",
#                 activation=tf.nn.relu)
#
#             tf.logging.info('conv1: ------> {}'.format(conv1))  # [?, 98, 40, 32]
#
#             # Pooling Layer #1
#             pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)  # [?, 49, 20, 32]
#             tf.logging.info('pool1: ------> {}'.format(pool1))
#
#             # Convolutional Layer #2 and Pooling Layer #2
#             conv2 = tf.layers.conv2d(
#                 inputs=pool1,
#                 filters=64,
#                 kernel_size=[3, 3],
#                 padding="same",
#                 activation=tf.nn.relu)
#             tf.logging.info('conv2: ------> {}'.format(conv2))  # [?, 49, 20, 64]
#
#             pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#             tf.logging.info('pool2: ------> {}'.format(pool2))  # [?, 24, 10, 64]
#
#             # Dense Layer
#             pool2_flat = tf.reshape(pool2, [-1, 24 * 10 * 64])
#             tf.logging.info('pool2_flat: ------> {}'.format(pool2_flat))  # [-1, 24 * 10 * 64]
#
#             dense = tf.layers.dense(inputs=pool2_flat, units=24 * 10 * 64, activation=tf.nn.relu)
#
#             hidden_layer = tf.layers.dropout(
#                 inputs=dense, rate=0.9, training=mode == tf.estimator.ModeKeys.TRAIN)
#
#             tf.logging.info('hidden_layer: ------> {}'.format(hidden_layer))
#
#         with  tf.name_scope("logits-layer"):
#             # [?, self.NUM_CLASSES]
#
#             logits = tf.layers.dense(inputs=hidden_layer,
#                                      units=self.sr_config._num_classes,
#                                      kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))
#
#             tf.logging.info('logits: ------> {}'.format(logits))
#
#         with  tf.name_scope("output-layer"):
#             # [?,1]
#             predicted_class = tf.argmax(logits, axis=1, name="class_output")
#             tf.logging.info('predicted_class: ------> {}'.format(predicted_class))
#
#             predicted_probabilities = tf.nn.softmax(logits, name="softmax_output")
#             tf.logging.info('predicted_probabilities: ------> {}'.format(predicted_probabilities))
#
#         predictions = {
#             "classes": predicted_class,
#             "probabilities": predicted_probabilities
#         }
#
#         tf.logging.info("=====> logits {}".format(logits))
#         tf.logging.info("=====> classes {}".format(predicted_class))
#
#         if mode != ModeKeys.INFER:
#             with tf.name_scope('cross_entropy'):
#                 losses = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
#             tf.logging.info("=====> losses {}".format(losses))
#
#
#
#         # Loss, training and eval operations are not needed during inference.
#         loss = None
#         train_op = None
#         eval_metric_ops = {}
#
#         if mode != ModeKeys.INFER:
#             with tf.name_scope('train-optimization'):#, tf.control_dependencies(control_dependencies):
#                 global_step = tf.train.get_global_step()
#                 learning_rate = self.sr_config._learning_rate
#                 train_op = tf.contrib.layers.optimize_loss(
#                     loss=losses,
#                     global_step=global_step,
#                     optimizer=tf.train.GradientDescentOptimizer,
#                     learning_rate=learning_rate)
#
#                 loss = losses
#
#             correct_prediction = tf.equal(predictions["classes"], labels)
#             confusion_matrix = tf.confusion_matrix(
#                 labels, predictions["classes"], num_classes=self.sr_config._label_count)
#
#             eval_metric_ops = {
#                 'Accuracy': tf.metrics.accuracy(
#                     labels=tf.cast(labels, tf.int32),
#                     predictions=predictions["classes"],
#                     name='accuracy'),
#                 'Precision': tf.metrics.precision(
#                     labels=tf.cast(labels, tf.int32),
#                     predictions=predictions["classes"],
#                     name='Precision'),
#                 'Recall': tf.metrics.recall(
#                     labels=tf.cast(labels, tf.int32),
#                     predictions=predictions["classes"],
#                     name='Recall')
#             }
#
#         return tf.estimator.EstimatorSpec(
#             mode=mode,
#             predictions=predictions,
#             loss=loss,
#             train_op=train_op,
#             eval_metric_ops=eval_metric_ops,
#             # training_hooks=self.hooks
#         )
