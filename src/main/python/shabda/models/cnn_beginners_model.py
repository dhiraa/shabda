# import tensorflow as tf
# from tensorflow.keras.layers import (Convolution1D, Dense,
#                                      Dropout, GlobalAveragePooling1D,
#                                      GlobalMaxPool1D, Input, MaxPool1D,
#                                      concatenate)
# from tensorflow.keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
#                                      GlobalMaxPool2D, MaxPool2D, concatenate, Activation)
#
# from tensorflow.contrib.learn import ModeKeys
#
# from shabda.helpers.print_helper import *
#
# class CustomDNN(tf.estimator.Estimator):
#     def __init__(self, model_dir, hparams, run_config):
#         super(CustomDNN, self).__init__(
#             model_fn=self._model_fn,
#             model_dir=model_dir,
#             config=run_config)
#
#
#     def _model_fn(self, features, labels, mode):
#         """Model function used in the estimator.
#
#         Args:
#             features : Tensor(shape=[?], dtype=string) Input features to the model.
#             labels : Tensor(shape=[?, n], dtype=Float) Input labels.
#             mode (ModeKeys): Specifies if training, evaluation or prediction.
#             params (HParams): hyperparameters.
#
#         Returns:
#             (EstimatorSpec): Model to be run by Estimator.
#         """
#         print_debug(features)
#
#         audio_mfcc = features["features"]
#
#         print_debug(audio_mfcc)
#         print_debug(features["labels"])
#
#         float_audio_features = tf.cast(audio_mfcc, tf.float32, name="float_audio_features")
#         audio_features = tf.identity(float_audio_features, name="audio_features")
#         features_reshaped = tf.reshape(audio_features, shape=(-1, 88200, 1), name="features_reshaped")
#
#
#         if mode != ModeKeys.INFER:
#             labels = features["labels"]
#             labels = tf.identity(labels, name="features")
#
#         tf.logging.debug('features -----> {}'.format(audio_features)) #shape=(?, 64, 256, 1)
#         tf.logging.debug('labels -----> {}'.format(labels))
#
#
#         with  tf.name_scope('conv_layer'):
#
#             # Convolutional Layer #1
#             conv1 = tf.layers.conv1d(
#                 inputs=features_reshaped,
#                 filters=16,
#                 kernel_size=9,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv1")
#
#             tf.logging.info('conv1: ------> {}'.format(conv1))  # (?, 64, 256, 32)
#
#             # Convolutional Layer #2
#             conv2 = tf.layers.conv1d(
#                 inputs=conv1,
#                 filters=16,
#                 kernel_size=9,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv2")
#
#             tf.logging.info('conv2: ------> {}'.format(conv2))  # (?, 64, 256, 32)
#
#             # Pooling Layer #1
#             pool1 = tf.layers.max_pooling1d(inputs=conv2,pool_size=16,strides=1 )  # (?, 16, 64, 32)
#             tf.logging.info('pool1: ------> {}'.format(pool1))
#
#             # Dropout1
#             if mode != ModeKeys.INFER:
#                 pool1 = tf.layers.dropout(inputs=pool1, rate=0.6, training=mode == tf.estimator.ModeKeys.TRAIN)
#
#                 tf.logging.info('drop1: ------> {}'.format(pool1))
#
#             # Convolutional Layer #3 and Pooling Layer #2
#             conv3 = tf.layers.conv1d(
#                 inputs=pool1,
#                 filters=32,
#                 kernel_size=3,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv3")
#             tf.logging.info('conv3: ------> {}'.format(conv3))  # (?, 16, 64, 64)
#
#             conv4 = tf.layers.conv1d(
#                 inputs=conv3,
#                 filters=32,
#                 kernel_size=3,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv4")
#             tf.logging.info('conv4: ------> {}'.format(conv4))  # (?, 16, 64, 64)
#
#             # Pooling Layer #2
#             pool2 = tf.layers.max_pooling1d(inputs=conv4, pool_size=4,strides=1)  # (?, 16, 64, 32)
#             tf.logging.info('pool2: ------> {}'.format(pool2))  # (?, 4, 16, 64)
#             # Dropout2
#             if mode != ModeKeys.INFER:
#                 pool2 = tf.layers.dropout(inputs=pool2, rate=0.6, training=mode == tf.estimator.ModeKeys.TRAIN)
#
#                 tf.logging.info('drop2: ------> {}'.format(pool2))
#
#             ################## Block 3
#             # Convolutional Layer #2 and Pooling Layer #2
#             conv5 = tf.layers.conv1d(
#                 inputs=pool2,
#                 filters=128,
#                 kernel_size=3,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv5")
#             tf.logging.info('conv5: ------> {}'.format(conv5))  # (?, 16, 64, 64)
#
#             conv6 = tf.layers.conv1d(
#                 inputs=conv5,
#                 filters=128,
#                 kernel_size=3,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv6")
#             tf.logging.info('conv6: ------> {}'.format(conv6))  # (?, 16, 64, 64)
#
#             # Pooling Layer #2
#             pool3 = tf.layers.max_pooling1d(inputs=conv6, pool_size=4,strides=1)  # (?, 16, 64, 32)
#             tf.logging.info('pool3: ------> {}'.format(pool3))  # (?, 4, 16, 64)
#
#             # Dropout2
#             if mode != ModeKeys.INFER:
#                 pool3 = tf.layers.dropout(inputs=pool3, rate=0.6, training=mode == tf.estimator.ModeKeys.TRAIN)
#
#                 tf.logging.info('drop3: ------> {}'.format(pool2))
#             ################ Block 4
#             # Convolutional Layer #7 and Pooling Layer 4
#             conv7 = tf.layers.conv1d(
#                 inputs=pool3,
#                 filters=256,
#                 kernel_size=3,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv7")
#             tf.logging.info('conv7: ------> {}'.format(conv7))  # (?, 16, 64, 64)
#
#             conv8 = tf.layers.conv1d(
#                 inputs=conv7,
#                 filters=256,
#                 kernel_size=3,
#                 padding="same",
#                 activation=tf.nn.relu,
#                 name="conv8")
#             tf.logging.info('conv8: ------> {}'.format(conv8))  # (?, 16, 64, 64)
#
#             # Pooling Layer #2
#             pool4 = tf.keras.layers.GlobalAveragePooling1D()(conv4)  # (?, 16, 64, 32)
#             tf.logging.info('pool4: ------> {}'.format(pool4))  # (?, 4, 16, 64)
#
#             # Dropout2
#             if mode != ModeKeys.INFER:
#                 pool4 = tf.layers.dropout(inputs=pool4, rate=0.6, training=mode == tf.estimator.ModeKeys.TRAIN)
#
#                 tf.logging.info('drop4: ------> {}'.format(pool4))
#             ##################
#             # Dense Layer
#             # pool_flat = tf.reshape(pool2, [-1, 4 * 16 * 64], name="pool_flat_reshape")
#             # tf.logging.info('pool4_flat: ------> {}'.format(pool_flat)) #60480
#
#             dense = tf.layers.dense(inputs=pool4, units=64, activation=tf.nn.relu)
#             tf.logging.info('dense_1: ------> {}'.format(dense))  # (?, 16, 64, 64)
#
#             dense = tf.layers.dense(inputs=dense, units=1028, activation=tf.nn.relu)
#             tf.logging.info('dense_@: ------> {}'.format(dense))  # (?, 16, 64, 64)
#
#
#         with  tf.name_scope("logits-layer"):
#             # [?, self.NUM_CLASSES]
#
#             logits = tf.layers.dense(inputs=dense,
#                                      units=41,
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
#         # logging
#         # self.log_tensors("output_probabilities", "output-layer/softmax_output")
#         # tf.summary.histogram(encoding.name, encoding)
#         tf.summary.histogram(predicted_probabilities.name, predicted_probabilities)
#
#         # Loss, training and eval operations are not needed during inference.
#         loss = None
#         train_op = None
#         eval_metric_ops = {}
#
#         if mode != ModeKeys.INFER:
#             tf.logging.info('labels: ------> {}'.format(labels))
#             tf.logging.info('predictions["classes"]: ------> {}'.format(predictions["classes"]))
#
#             loss = tf.losses.softmax_cross_entropy(
#                 onehot_labels=labels,
#                 logits=logits,
#                 weights=0.80,
#                 scope='actual_loss')
#
#             loss = tf.reduce_mean(loss, name="reduced_mean")
#
#             train_op = tf.contrib.layers.optimize_loss(
#                 loss=loss,
#                 global_step=tf.train.get_global_step(),
#                 optimizer=tf.train.AdamOptimizer,
#                 learning_rate=0.001)
#
#             label_argmax = tf.argmax(labels, 1, name='label_argmax')
#
#             eval_metric_ops = {
#                 'Accuracy': tf.metrics.accuracy(
#                     labels=label_argmax,
#                     predictions=predictions["classes"],
#                     name='accuracy'),
#                 'Precision': tf.metrics.precision(
#                     labels=label_argmax,
#                     predictions=predictions["classes"],
#                     name='Precision'),
#                 'Recall': tf.metrics.recall(
#                     labels=label_argmax,
#                     predictions=predictions["classes"],
#                     name='Recall')
#             }
#             tf.summary.scalar(loss.name, loss)
#
#         return tf.estimator.EstimatorSpec(
#             mode=mode,
#             predictions=predictions,
#             loss=loss,
#             train_op=train_op,
#             eval_metric_ops=eval_metric_ops
#         )
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
