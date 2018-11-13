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
Utilities for audio related operations.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


# import re
# import os
# from glob import glob
# import librosa
# from scipy.io import wavfile

def pad_data_array(data, max_audio_length):
    """
    To pad the audio data with zeros
    :param data:
    :param max_audio_length:
    :return:
    """
    # Random offset / Padding
    if len(data) > max_audio_length:
        max_offset = len(data) - max_audio_length
        offset = np.random.randint(max_offset)
        data = data[offset:(max_audio_length + offset)]
    else:
        if max_audio_length > len(data):
            max_offset = max_audio_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, max_audio_length - len(data) - offset), "constant")
    return data


def audio_norm(data):
    """
    Normalizes the input np.array
    :param data: np.array
    :return: np.array
    """
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 1e-6)
    return data - 0.5

# def get_frequency_spectrum(data, sampling_rate, n_mfcc):
#     """
#
#     :param data:
#     :param sampling_rate:
#     :param n_mfcc:
#     :return:
#     """
#     return librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc, n_fft=1024,
#                                 hop_length=345)  # with sr=44KHz and hop_length=345 to get n_mfcc x 256
#
#
# def load_wav_audio_file(file_path):
#     """
#
#     :param file_path:
#     :return:
#     """
#     data, sample_rate = librosa.core.load(file_path, sr=None, res_type="kaiser_fast")
#     return data, sample_rate


# def prepare_words_list(possible_speech_commands,
#                        silence_lable="_silence_",
#                        unknown_word_label="_unknown_"):
#     """Prepends common tokens to the custom word list.
#     Args:
#       possible_speech_commands: List of strings containing the custom words.
#     Returns:
#       List with the standard silence and unknown tokens added.
#     """
#     return [silence_lable, unknown_word_label] + possible_speech_commands

# def prepare_audio_sampling_settings(label_count,
#                                     sample_rate,
#                                     clip_duration_ms,
#                                     window_size_ms,
#                                     window_stride_ms,
#                                     dct_coefficient_count):
#   """Calculates common settings needed for all models.
#   Args:
#     label_count: How many classes are to be recognized.
#     sample_rate: Number of audio samples per second. (Default: 16000)
#     clip_duration_ms: Length of each audio clip to be analyzed. (Default: 1000)
#     window_size_ms: Duration of frequency analysis window. (Default: 30)
#     window_stride_ms: How far to move in time between frequency windows. (Default: 10)
#     dct_coefficient_count: Number of frequency bins to use for analysis. (Default: 40)
#   Returns:
#     Dictionary containing common settings.
#   """
#   desired_samples = int(sample_rate * clip_duration_ms / 1000) # 16000
#   window_size_samples = int(sample_rate * window_size_ms / 1000) # 16000 * 30 / 1000 = 180
#   window_stride_samples = int(sample_rate * window_stride_ms / 1000) # 16000 * 10 / 1000 = 160
#   length_minus_window = (desired_samples - window_size_samples) # 16000 - 180 = 15020
#
#   if length_minus_window < 0: spectrogram_length = 0
#   else: spectrogram_length = 1 + int(length_minus_window / window_stride_samples) # 1 + (15020/160) = 94
#
#   fingerprint_size = dct_coefficient_count * spectrogram_length # 40 * 94 = 3760
#
#   return {
#       'desired_samples': desired_samples,
#       'window_size_samples': window_size_samples,
#       'window_stride_samples': window_stride_samples,
#       'spectrogram_length': spectrogram_length,
#       'dct_coefficient_count': dct_coefficient_count,
#       'fingerprint_size': fingerprint_size,
#       'label_count': label_count,
#       'sample_rate': sample_rate,
#   }

#
# DATA_DIR = "../data/tensorflow_speech_recoginition_challenge/"
# OUT_DIR = "tensorflow_speech_recoginition_challenge/"
# paths = glob(os.path.join(DATA_DIR, 'test/audio/*wav'))
#
# POSSIBLE_LABELS = 'yes no up down left right on off stop go unknown'.split()
# id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
# name2id = {name: i for i, name in id2name.items()}
#
#
# def load_data(data_dir):
#     '''
#     trainset, valset = load_data(DATA_DIR)
#
#     :param data_dir:
#     :return:
#     '''
#     """ Return 2 lists of tuples:
#     [(class_id, user_id, path), ...] for train
#     [(class_id, user_id, path), ...] for validation
#     """
#     # Just a simple regexp for paths with three groups:
#     # prefix, label, user_id
#     pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
#
#     all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))
#
#     with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
#         validation_files = fin.readlines()
#
#     valset = set()
#     for entry in validation_files:
#         r = re.match(pattern, entry)
#         if r:
#             valset.add(r.group(3))
#
#     possible = set(POSSIBLE_LABELS)
#     train, val = [], []
#
#     for entry in all_files:
#         r = re.match(pattern, entry)
#         if r:
#             label, uid = r.group(2), r.group(3)
#             if label == '_background_noise_':
#                 label = 'silence'
#             if label not in possible:
#                 label = 'unknown'
#
#             label_id = name2id[label]
#
#             sample = (label_id, uid, entry)
#
#             if uid in valset:
#                 val.append(sample)
#             else:
#                 train.append(sample)
#
#     print('There are {} train and {} val samples'.format(len(train), len(val)))
#     return train, val
#
#
# def data_generator(data, params, mode='train'):
#     def generator():
#         if mode == 'train':
#             np.random.shuffle(data)
#         # Feel free to add any augmentation
#         for (label_id, uid, fname) in data:
#             try:
#                 _, wav = wavfile.read(fname)
#                 wav = wav.astype(np.float32) / np.iinfo(np.int16).max
#
#                 L = 16000  # be aware, some files are shorter than 1 sec!
#                 if len(wav) < L:
#                     continue
#                 # let's generate more silence!
#                 samples_per_file = 1 if label_id != name2id['silence'] else 20
#                 for _ in range(samples_per_file):
#                     if len(wav) > L:
#                         beg = np.random.randint(0, len(wav) - L)
#                     else:
#                         beg = 0
#                     yield dict(
#                         target=np.int32(label_id),
#                         wav=wav[beg: beg + L],
#                     )
#             except Exception as err:
#                 print(err, label_id, uid, fname)
#
#     return generator
