import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys

from shabda.helpers.print_helper import *

class CustomDNN(tf.estimator.Estimator):
    def __init__(self, model_dir, run_config, hparams=None):
        super(CustomDNN, self).__init__(
            model_fn=self._model_fn,
            model_dir=model_dir,
            config=run_config)


    def defaul_params(self):
        config = {

        }

        return config

    def _model_fn(self, features, labels, mode):
        """Model function used in the estimator.

        Args:
            features : Tensor(shape=[?], dtype=string) Input features to the model.
            labels : Tensor(shape=[?, n], dtype=Float) Input labels.
            mode (ModeKeys): Specifies if training, evaluation or prediction.
            params (HParams): hyperparameters.

        Returns:
            (EstimatorSpec): Model to be run by Estimator.
        """
        print_debug(features)

        audio_mfcc = features["features"]

        print_debug(audio_mfcc)
        print_debug(features["labels"])

        float_audio_features = tf.cast(audio_mfcc, tf.float32, name="float_audio_features")
        audio_features = tf.identity(float_audio_features, name="audio_features")
        features_reshaped = tf.reshape(audio_features, shape=(-1, 64, 256, 1), name="features_reshaped")


        if mode != ModeKeys.INFER:
            labels = features["labels"]
            labels = tf.identity(labels, name="features")

        tf.logging.debug('features -----> {}'.format(audio_features)) #shape=(?, 64, 256, 1)
        tf.logging.debug('labels -----> {}'.format(labels))


        with  tf.name_scope('conv_layer'):

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
            tf.logging.info('pool3_flat: ------> {}'.format(pool_flat)) #60480

            dense = tf.layers.dense(inputs=pool_flat, units=512, activation=tf.nn.relu)

            # if mode != ModeKeys.INFER:
            #     dense = tf.layers.dropout(
            #         inputs=dense, rate=0.9, training=mode == tf.estimator.ModeKeys.TRAIN)
            #
            #     tf.logging.info('dense_layer: ------> {}'.format(dense))

        with  tf.name_scope("logits-layer"):
            # [?, self.NUM_CLASSES]

            logits = tf.layers.dense(inputs=dense,
                                     units=41,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))

            tf.logging.info('logits: ------> {}'.format(logits))

        with  tf.name_scope("output-layer"):
            # [?,1]
            predicted_class = tf.argmax(logits, axis=1, name="class_output")
            tf.logging.info('predicted_class: ------> {}'.format(predicted_class))

            predicted_probabilities = tf.nn.softmax(logits, name="softmax_output")
            tf.logging.info('predicted_probabilities: ------> {}'.format(predicted_probabilities))

        predictions = {
            "classes": predicted_class,
            "probabilities": predicted_probabilities
        }

        # logging
        # self.log_tensors("output_probabilities", "output-layer/softmax_output")
        # tf.summary.histogram(encoding.name, encoding)
        tf.summary.histogram(predicted_probabilities.name, predicted_probabilities)

        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}

        if mode != ModeKeys.INFER:
            tf.logging.info('labels: ------> {}'.format(labels))
            tf.logging.info('predictions["classes"]: ------> {}'.format(predictions["classes"]))

            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=labels,
                logits=logits,
                weights=0.80,
                scope='actual_loss')

            loss = tf.reduce_mean(loss, name="reduced_mean")

            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                optimizer=tf.train.AdamOptimizer,
                learning_rate=0.001)

            label_argmax = tf.argmax(labels, 1, name='label_argmax')

            eval_metric_ops = {
                'Accuracy': tf.metrics.accuracy(
                    labels=label_argmax,
                    predictions=predictions["classes"],
                    name='accuracy'),
                'Precision': tf.metrics.precision(
                    labels=label_argmax,
                    predictions=predictions["classes"],
                    name='Precision'),
                'Recall': tf.metrics.recall(
                    labels=label_argmax,
                    predictions=predictions["classes"],
                    name='Recall')
            }
            tf.summary.scalar(loss.name, loss)
            # validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            #     test_set.audio_utils,
            #     test_set.target,
            #     every_n_steps=50,
            #     metrics=validation_metrics,
            #     early_stopping_metric="loss",
            #     early_stopping_metric_minimize=True,
            #     early_stopping_rounds=200)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )



#     def _model_fn(self, features, labels, mode):
#         numeric_features = features["mfcc"]
#
#         tf.logging.debug('token_ids -----> {}'.format(numeric_features))
#         tf.logging.debug('labels -----> {}'.format(labels))
#
#         input_layer = tf.identity(numeric_features, name="input")
#
#         inp = Input(shape=(config.dim[0], config.dim[1], 1))
#         x = Convolution2D(32, (4, 10), padding="same")(inp)
#         x = BatchNormalization()(x)
#         x = Activation("relu")(x)
#         x = MaxPool2D()(x)
#
#         x = Convolution2D(32, (4, 10), padding="same")(x)
#         x = BatchNormalization()(x)
#         x = Activation("relu")(x)
#         x = MaxPool2D()(x)
#
#         x = Convolution2D(32, (4, 10), padding="same")(x)
#         x = BatchNormalization()(x)
#         x = Activation("relu")(x)
#         x = MaxPool2D()(x)
#
#         x = Convolution2D(32, (4, 10), padding="same")(x)
#         x = BatchNormalization()(x)
#         x = Activation("relu")(x)
#         x = MaxPool2D()(x)
#
#         x = Flatten()(x)
#         x = Dense(64)(x)
#         x = BatchNormalization()(x)
#         x = Activation("relu")(x)
#         out = Dense(nclass, activation=softmax)(x)
#
#         # Loss, training and eval operations are not needed during inference.
#         loss = None
#         train_op = None
#         eval_metric_ops = {}
#
#         if mode != ModeKeys.INFER:
#             loss = tf.sqrt(tf.reduce_mean(tf.square(y_ - labels), name="reduced_mean"))
# #             loss =  tf.sqrt(tf.losses.mean_squared_error(y_, labels), name="reduced_mean")
#
#             train_op = tf.contrib.layers.optimize_loss(
#                                                     loss=loss,
#                                                     global_step=tf.train.get_global_step(),
#                                                     optimizer=tf.train.RMSPropOptimizer,
#                                                     learning_rate=0.001)
#             eval_metric_ops = {
#                 "rmse_loss" : tf.metrics.root_mean_squared_error(labels=labels, predictions=y_)
#             }
#
#
#         return tf.estimator.EstimatorSpec(
#             mode=mode,
#             predictions=y_,
#             loss=loss,
#             train_op=train_op,
#             eval_metric_ops=eval_metric_ops
#         )
#
