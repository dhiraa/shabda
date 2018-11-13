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
Naive LSTM Audio classifier
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from overrides import overrides
from shabda.models.clasifiers.classifer_base import ClassifierBase


class NaiveLSTM(ClassifierBase):
    def __init__(self, hparams=None):
        super(NaiveLSTM, self).__init__(hparams=hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        hparams = {
            "name": "lstm_naive",
            "out_dim": -1,
            "learning_rate": 0.001
        }
        return hparams

    @overrides
    def _build_layers(self, features, mode):
        # features = tf.cast(features, tf.float32, name="float_audio_features")

        is_training = tf.estimator.ModeKeys.PREDICT != mode

        tf.logging.info('features -----> {}'.format(features))

        with  tf.name_scope("lstm_layer"):
            # Create a LSTM Unit cell with hidden size of EMBEDDING_SIZE.
            d_rnn_cell_fw_one = tf.nn.rnn_cell.LSTMCell(32,
                                                        state_is_tuple=True)
            d_rnn_cell_bw_one = tf.nn.rnn_cell.LSTMCell(32,
                                                        state_is_tuple=True)

            if is_training:
                d_rnn_cell_fw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_fw_one,
                                                                  output_keep_prob=0.5)
                d_rnn_cell_bw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_bw_one,
                                                                  output_keep_prob=0.5)
            else:
                d_rnn_cell_fw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_fw_one, output_keep_prob=1.0)
                d_rnn_cell_bw_one = tf.contrib.rnn.DropoutWrapper(d_rnn_cell_bw_one, output_keep_prob=1.0)

            d_rnn_cell_fw_one = tf.nn.rnn_cell.MultiRNNCell(cells=[d_rnn_cell_fw_one] *
                                                                  1,
                                                            state_is_tuple=True)
            d_rnn_cell_bw_one = tf.nn.rnn_cell.MultiRNNCell(cells=[d_rnn_cell_bw_one] *
                                                                  1,
                                                            state_is_tuple=True)

            tf.logging.info('d_rnn_cell_fw_one -----> {}'.format(d_rnn_cell_fw_one))
            tf.logging.info('d_rnn_cell_bw_one -----> {}'.format(d_rnn_cell_bw_one))

            (fw_output_one, bw_output_one), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=d_rnn_cell_fw_one,
                cell_bw=d_rnn_cell_bw_one,
                dtype=tf.float64,
                inputs=features,
                scope="encode")

            # [BATCH_SIZE, MAX_SEQ_LENGTH, 2*WORD_LEVEL_LSTM_HIDDEN_SIZE) TODO check MAX_SEQ_LENGTH?
            bilstm_out = tf.concat([fw_output_one,
                                    bw_output_one], axis=-1, name="fw_bw")

            tf.logging.info('bilstm_out -----> {}'.format(bilstm_out))

            encoded_data = tf.reshape(bilstm_out, shape=[-1, 128 * 64])

            tf.logging.info('encoded_data -----> {}'.format(encoded_data))

            combined_logits = tf.layers.dense(inputs=encoded_data,
                                              units=self._out_dim * 10,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42),
                                              activation=tf.nn.relu)

            combined_logits = tf.layers.dense(inputs=combined_logits,
                                              units=self._out_dim,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(seed=42))

            tf.logging.info('combined_logits: ------> {}'.format(combined_logits))

            return combined_logits
