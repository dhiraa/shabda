import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import signal
# it's a magic function :)
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
from tqdm import tqdm
from sarvam.helpers.print_helper import *
from speech_recognition.dataset.feature_types import RawWavAudio
from speech_recognition.sr_config.sr_config import *


class SimpleSpeechRecognizerConfig:
    def __init__(self):
        self._model_dir = "experiments/SimpleSpeechRecognizer/"

        self.SEED = 2018
        self.BATCH_SIZE = 64
        self.KEEP_PROB = 0.5
        self.LEARNING_RATE = 1e-3
        self.CLIP_GRADIENTS = 15.0
        self.USE_BATCH_NORM = True
        self.NUM_CLASSES = len(POSSIBLE_COMMANDS) + 2

    @staticmethod
    def user_config():
        return SimpleSpeechRecognizerConfig()


class SimpleSpeechRecognizer(tf.estimator.Estimator):
    def __init__(self,
                 sr_config: SimpleSpeechRecognizerConfig, run_config):
        super(SimpleSpeechRecognizer, self).__init__(
            model_fn=self._model_fn,
            model_dir=sr_config._model_dir,
            config=run_config)

        self.sr_config = sr_config
        self._feature_type = RawWavAudio

    def baseline(self, x, params, is_training):

        x = layers.batch_norm(x, is_training=is_training)

        print_error("x =====> {}\n".format(x))

        for i in range(4):
            x = layers.conv2d(
                x, 16 * (2 ** i), 3, 1,
                activation_fn=tf.nn.elu,
                normalizer_fn=layers.batch_norm if self.sr_config.USE_BATCH_NORM else None,
                normalizer_params={'is_training': is_training}
            )
            x = layers.max_pool2d(x, 2, 2)

        print_error("x_mpool =====> {}\n".format(x))

        # just take two kind of pooling and then mix them, why not :)
        mpool = tf.reduce_max(x, axis=[1, 2], keep_dims=True)
        apool = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

        x = 0.5 * (mpool + apool)

        # we can use conv2d 1x1 instead of dense
        x = layers.conv2d(x, 128, 1, 1, activation_fn=tf.nn.elu)
        print_error("x_logits =====> {}\n".format(x))

        x = tf.nn.dropout(x, keep_prob=self.sr_config.KEEP_PROB if is_training else 1.0)


        # again conv2d 1x1 instead of dense layer
        logits = layers.conv2d(x, self.sr_config.NUM_CLASSES, 1, 1, activation_fn=None)

        print_error("x_logits =====> {}\n".format(x))

        return tf.squeeze(logits, [1, 2])


    # features is a dict with keys: tensors from our datagenerator
    # labels also were in features, but excluded in generator_input_fn by target_key

    def _model_fn(self, features, labels, mode, params, config):

        control_dependencies = []
        checks = tf.add_check_numerics_ops()
        control_dependencies = []


        # wav is a waveform signal with shape (16000, )
        wav = features[self._feature_type.FEATURE_1]
        wav = tf.cast(wav, tf.float32)

        tf.logging.info("wav =====> {}\n".format(wav))

        print_error("wav =====> {}\n".format(wav))

        tf.logging.info("labels =====> {}\n".format(labels))

        print_error("labels =====> {}\n".format(labels))

        # we want to compute spectograms by means of short time fourier transform:
        specgram = signal.stft(
            wav,
            400,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
            160,  # 16000 * 0.010 -- default stride
        )
        # specgram is a complex tensor, so split it into abs and phase parts:
        phase = tf.angle(specgram) / np.pi
        # log(1 + abs) is a default transformation for energy units
        amp = tf.log1p(tf.abs(specgram))

        x = tf.stack([amp, phase], axis=3)  # shape is [bs, time, freq_bins, 2]
        x = tf.to_float(x)  # we want to have float32, not float64

        print_error("x =====> {}\n".format(x))
        # Im really like to use make_template instead of variable_scopes and re-usage
        extractor = tf.make_template(
            'extractor',
            self.baseline,
            create_scope_now_=True,
        )

        logits = extractor(x, params, mode == tf.estimator.ModeKeys.TRAIN)

        if mode == tf.estimator.ModeKeys.TRAIN:
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

            # some lr tuner, you could use move interesting functions
            def learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=self.sr_config.LEARNING_RATE,
                optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
                learning_rate_decay_fn=learning_rate_decay_fn,
                clip_gradients=self.sr_config.CLIP_GRADIENTS,
                variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

            specs = dict(
                mode=mode,
                loss=loss,
                train_op=train_op,
            )

        if mode == tf.estimator.ModeKeys.EVAL:
            prediction = tf.argmax(logits, axis=-1)
            acc, acc_op = tf.metrics.mean_per_class_accuracy(
                labels, prediction, self.sr_config.NUM_CLASSES)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            specs = dict(
                mode=mode,
                loss=loss,
                eval_metric_ops=dict(
                    acc=(acc, acc_op),
                )
            )

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'label': tf.argmax(logits, axis=-1),  # for probability just take tf.nn.softmax()
                'sample': features['sample'],  # it's a hack for simplicity
            }
            specs = dict(
                mode=mode,
                predictions=predictions,
            )
        return tf.estimator.EstimatorSpec(**specs)