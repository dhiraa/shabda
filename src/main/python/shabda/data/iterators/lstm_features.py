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
Iterator that creates features for LSTM based models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import librosa
import tensorflow as tf
from overrides import overrides

from shabda.data.dataset.freesound_dataset import FreeSoundAudioDataset
from shabda.data.iterators.internal.free_sound_data_iterator_base import FreeSoundDataIteratorBase

class LSTMFeatureIterator(FreeSoundDataIteratorBase):
    def __init__(self, hparams, dataset: FreeSoundAudioDataset):
        super(LSTMFeatureIterator, self).__init__(hparams, dataset)

        if not isinstance(dataset, FreeSoundAudioDataset):
            raise AssertionError("dataset should be FreeSoundAudioDataset")

    def append_zeros(self, required_dim, data, is_row=False):
        """

        :param required_dim:
        :param data:
        :param is_row:
        :return:
        """
        rows, cols = data.shape

        new_rows = required_dim - rows
        new_cols = required_dim - cols
        if is_row:
            if rows > required_dim:
                return data
            return np.vstack([data, np.zeros(shape=(new_rows, cols))])
        else:
            if cols > required_dim:
                return data
            return np.hstack([data, np.zeros(shape=(rows, new_cols))])

    # References: https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification
    def extract_audio_features(self, audio_file, hop_length,  timeseries_length = 128):
        """
        Combine 4 audio features as one time series feature, namely : MFCC, Spectral Centroid,
        Chroma STFT, Spectral Contrast
        :param audio_file: Full path of the audio file
        :param hop_length:
        :return: [timeseries_length, 33]
        """
        # timeseries_length = min(self.timeseries_length_list)

        data = np.zeros((1, timeseries_length, 33), dtype=np.float64)
        target = []

        y, sr = librosa.load(audio_file, duration=3.0)
        y, index = librosa.effects.trim(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        mfcc = self.append_zeros(timeseries_length, mfcc)
        # print(mfcc.T.shape)

        spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
        spectral_center = self.append_zeros(timeseries_length, spectral_center)
        # print(spectral_center.T.shape)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        chroma = self.append_zeros(timeseries_length, chroma)
        # print(chroma.T.shape)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        spectral_contrast = self.append_zeros(timeseries_length, spectral_contrast)
        # print(spectral_contrast.T.shape)

        data[:, :, 0:13] = mfcc.T[0:timeseries_length, :]
        data[:, :, 13:14] = spectral_center.T[0:timeseries_length, :]
        data[:, :, 14:26] = chroma.T[0:timeseries_length, :]
        data[:, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

        #print("Extracted features audio track %s." % audio_file)

        return data  # , np.expand_dims(np.asarray(target), axis=1)

    @overrides
    def _user_map_func(self, file_path, label):
        """
        Function that maps the audio files into features and labels as on-hot vector
        :param file_path:
        :param label:
        :return:
        """
        data = self.extract_audio_features(audio_file=file_path, hop_length=512)

        label = self._dataset.get_one_hot_encoded(label)
        return data, label

    @overrides
    def _user_resize_func(self, data, label):
        """
        Function that sets up the sizes of the tensor, after execution of `tf.py_func` call
        :param data:
        :param label:
        :return:
        """
        data = tf.reshape(data, shape=[128,33])
        label = tf.reshape(label, shape=[42])
        return data, label




