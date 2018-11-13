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
An iterator that produces MFCC features
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import librosa
from overrides import overrides

from shabda.data.dataset.freesound_dataset import FreeSoundAudioDataset
from shabda.data.iterators.internal.free_sound_data_iterator_base import FreeSoundDataIteratorBase
from shabda.helpers import audio
from shabda.helpers.print_helper import *


class MFCCDataIterator(FreeSoundDataIteratorBase):
    def __init__(self, hparams, dataset: FreeSoundAudioDataset):
        super(MFCCDataIterator, self).__init__(hparams, dataset)

        if not isinstance(dataset, FreeSoundAudioDataset):
            raise AssertionError("dataset should be FreeSoundAudioDataset")

    def get_frequency_spectrum(self, data, sampling_rate, n_mfcc):
        """

        :param data:
        :param sampling_rate:
        :param n_mfcc:
        :return:
        """
        return librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc, n_fft=1024,
                                    hop_length=345)  # with sr=44KHz and hop_length=345 to get n_mfcc x 256

    def load_wav_audio_file(self, file_path):
        """

        :param file_path:
        :return:
        """
        data, sample_rate = librosa.core.load(file_path, sr=None, res_type="kaiser_fast")
        return data, sample_rate

    @overrides
    def _user_map_func(self, file_path, label):
        """
        Function that maps the audio files into features and labels as on-hot vector
        :param file_path:
        :param label:
        :return:
        """
        data, sample_rate = self.load_wav_audio_file(file_path=file_path)

        data = audio.pad_data_array(data=data,
                                    max_audio_length=self._max_audio_length)

        data = self.get_frequency_spectrum(data=data, n_mfcc=self._n_mfcc,
                                           sampling_rate=self._sampling_rate)

        data = np.expand_dims(data, axis=-1)  # to make it compatible with CNN network
        data = data.flatten(order="C")  # row major

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
        return data, label
