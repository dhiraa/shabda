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
FreeSound Data Iterator base class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from  overrides import overrides

from shabda.data.iterators.internal.data_iterator_base import DataIteratorBase
from shabda.data.dataset.internal.audio_dataset_base import AudioDatasetBase
from shabda.helpers.print_helper import *

class FreeSoundDataIteratorBase(DataIteratorBase):
    """
    FreeSound Data set iterators which makes use of `tf.data` APIs
    """
    def __init__(self, hparams, dataset: AudioDatasetBase):
        DataIteratorBase.__init__(self, hparams, dataset)

        if not isinstance(dataset,  AudioDatasetBase):
            raise AssertionError("dataser should be an inherited class of AudioDatasetBase")

        self.name = "FreeSoundDataIteratorBase"

        self._batch_size = self._hparams.batch_size
        self._num_epochs = self._hparams.num_epochs
        self._sampling_rate = self._hparams.sampling_rate
        self._max_audio_length = self._hparams.sampling_rate * self._hparams.audio_duration  # 32000
        self._n_mfcc = self._hparams.n_mfcc

        self._train_files_path = dataset.get_train_files()
        self._train_labels = dataset.get_train_labels()

        self._val_files_path = dataset.get_val_files()
        self._val_labels = dataset.get_val_labels()

        self.test_files_path = []

    @staticmethod
    def get_default_params():
        config = {"use_mfcc": False,
        "n_mfcc": 64,
        "batch_size": 64,
        "num_epochs" : 1,
        "sampling_rate": 44100,
        "audio_duration": 2}

        return config

    def _user_map_func(self, audio_file_path, label):
        raise NotImplementedError

    def _user_resize_func(self, audio_file_path, label):
        raise NotImplementedError

    @overrides
    def _get_train_input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((self._train_files_path, self._train_labels))
        dataset = dataset.map(
            lambda filename, label: tuple(tf.py_func(
                self._user_map_func, [filename, label], [tf.double, tf.int64])),
            num_parallel_calls=4)
        # dataset = dataset.shuffle(len(self._train_labels))
        # dataset.prefetch()
        # dataset = dataset.repeat(self._num_epochs)
        dataset = dataset.map(self._user_resize_func, num_parallel_calls=4)
        dataset = dataset.prefetch(self._batch_size*2)
        dataset = dataset.batch(self._batch_size)
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)


        return dataset

    @overrides
    def _get_val_input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((self._val_files_path, self._val_labels))
        dataset = dataset.map(
            lambda filename, label: tuple(tf.py_func(
                self._user_map_func, [filename, label], [tf.double, label.dtype])),
            num_parallel_calls=4)
        # dataset = dataset.shuffle(len(self._val_labels))
        # dataset = dataset.repeat(self._num_epochs)
        dataset = dataset.map(self._user_resize_func, num_parallel_calls=4)

        dataset = dataset.prefetch(self._batch_size*2)
        dataset = dataset.batch(self._batch_size)

        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)

        return dataset


    @overrides
    def _get_test_input_function(self):
        raise NotImplementedError