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

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import SpectrogramProcessor, LogarithmicFilteredSpectrogramProcessor
from madmom.audio.filters import LogFilterbank
from madmom.processors import SequentialProcessor, Processor

from shabda.data.dataset.freesound_dataset import FreeSoundAudioDataset
from shabda.data.iterators.internal.free_sound_data_iterator_base import FreeSoundDataIteratorBase


class MadmomFeatureIteratorV2(FreeSoundDataIteratorBase):
    """
    Custom feature extraction using Madmom library pipepline
    Reference: https://github.com/CPJKU/dcase_task2/blob/master/dcase_task2/prepare_spectrograms.py
    """

    def __init__(self, hparams, dataset: FreeSoundAudioDataset):
        super(MadmomFeatureIteratorV2, self).__init__(hparams, dataset)

        if not isinstance(dataset, FreeSoundAudioDataset):
            raise AssertionError("dataset should be FreeSoundAudioDataset")

        sig_proc = SignalProcessor(num_channels=1, sample_rate=32000, norm=True)
        fsig_proc = FramedSignalProcessor(frame_size=1024, hop_size=128, origin='future')
        spec_proc = SpectrogramProcessor(frame_size=1024)
        filt_proc = LogarithmicFilteredSpectrogramProcessor(filterbank=LogFilterbank, num_bands=26, fmin=20, fmax=14000)
        processor_pipeline2 = [sig_proc, fsig_proc, spec_proc, filt_proc]
        self.processor_version2 = SequentialProcessor(processor_pipeline2)

    @overrides
    def _user_map_func(self, file_path, label):
        """
        Function that maps the audio files into features and labels as on-hot vector
        :param file_path:
        :param label:
        :return:
        """
        data = self.processor_version2.process(file_path)

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
