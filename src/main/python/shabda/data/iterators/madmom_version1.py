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
Iterator that creates features using madmom
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import os
import time
import glob
import librosa
import numpy as np
import tensorflow as tf
from overrides import overrides
import matplotlib.pyplot as plt

from madmom.processors import Processor

from shabda.data.dataset.freesound_dataset import FreeSoundAudioDataset
from shabda.data.iterators.internal.free_sound_data_iterator_base import FreeSoundDataIteratorBase


class LibrosaProcessor(Processor):

    def __init__(self):
        pass

    def process(self, file_path, **kwargs):
        n_fft = 1024
        sr = 32000
        mono = True
        log_spec = False
        n_mels = 128

        hop_length = 192
        fmax = None

        if mono:
            sig, sr = librosa.load(file_path, sr=sr, mono=True)
            sig = sig[np.newaxis]
        else:
            sig, sr = librosa.load(file_path, sr=sr, mono=False)
            # sig, sf_sr = sf.read(file_path)
            # sig = np.transpose(sig, (1, 0))
            # sig = np.asarray([librosa.resample(s, sf_sr, sr) for s in sig])

        spectrograms = []
        for y in sig:

            # compute stft
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                                pad_mode='reflect')

            # keep only amplitures
            stft = np.abs(stft)

            # spectrogram weighting
            if log_spec:
                stft = np.log10(stft + 1)
            else:
                freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
                stft = librosa.perceptual_weighting(stft ** 2, freqs, ref=1.0, amin=1e-10, top_db=99.0)

            # apply mel filterbank
            spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)

            # keep spectrogram
            spectrograms.append(np.asarray(spectrogram))

        spectrograms = np.asarray(spectrograms)

        return spectrograms


class MadmomFeatureIteratorV1(FreeSoundDataIteratorBase):
    """
    Custom melspectrogram feature extraction using Madmom library pipepline
    using librosa APIs
    Reference: https://github.com/CPJKU/dcase_task2/blob/master/dcase_task2/prepare_spectrograms.py
    """

    def __init__(self, hparams, dataset: FreeSoundAudioDataset):
        super(MadmomFeatureIteratorV1, self).__init__(hparams, dataset)

        if not isinstance(dataset, FreeSoundAudioDataset):
            raise AssertionError("dataset should be FreeSoundAudioDataset")

        self.processor_version1 = LibrosaProcessor()

    @overrides
    def _user_map_func(self, file_path, label):
        """
        Function that maps the audio files into features and labels as on-hot vector
        :param file_path:
        :param label:
        :return:
        """
        data = self.processor_version1.process(file_path)

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
        data = tf.reshape(data, shape=[128, 33])
        label = tf.reshape(label, shape=[42])
        return data, label
