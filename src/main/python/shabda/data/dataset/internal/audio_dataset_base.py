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
Interface class for all Audio Datasets
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import json
import os
import numpy as np

from shabda.hyperparams.hyperparams import HParams
# TODO: dont use `from ... import *`
from shabda.helpers.print_helper import *


class AudioDatasetBase(object):
    """
    Interface class for Audio Datasets.

    Any audio dataset is expected to inherit this class and give implementation.
    AudioDataset expectations are:
    - Provide list of training, validation and test files files
    - Expose list of lables and predefined labels, which then can be used for label indexing
    """
    def __init__(self, hparams, unknown_label="_unknown_"):
        self._hparams = HParams(hparams, self.default_hparams())
        self.is_init = False

        self._labels = None
        self._labels_2_index = None
        self._index_2_labels = None
        self._labels_dim = None
        self._unknown_label = unknown_label

    def init(self):
        """
        Inheriting class must take care of implementing this method
        Should call self._setup_labels() along with other initializations
        :return: None
        """
        self.is_init = True
        raise NotImplementedError

    @staticmethod
    def default_hparams():
        """
        Exposes the default params for this module
        :return: dict: params
        """
        params = {
            "labels_index_map_store_path" : "/tmp/shabda/"
        }
        return params

    def get_labels_dim(self):
        """
        Returns the total number of labels in this dataset along with any deafault lables
        like silent, back_ground_noise, if any.
        :return: int
        """
        return self._labels_dim

    def get_dataset_name(self):
        """
        Inheriting class must take care of implementing this method
        :return: Name of the dataset
        """
        raise NotImplementedError

    def get_num_train_samples(self):
        """
        Inheriting class must take care of implementing this method
        :return: int Number of training sample for current dataset
        """
        raise NotImplementedError

    def get_train_files(self):
        """
        Inheriting class must take care of implementing this method
        :return: list of files for training
        """
        raise NotImplementedError

    def get_train_labels(self):
        """
        Inheriting class must take care of implementing this method
        :return: list of labels in sync with training files
        """
        raise NotImplementedError

    def get_val_files(self):
        """
        Inheriting class must take care of implementing this method
        :return: list of validation files
        """
        raise NotImplementedError

    def get_val_labels(self):
        """
        Inheriting class must take care of implementing this method
        :return: list of labels in sync with validation files
        """
        raise NotImplementedError

    def get_test_files(self):
        """
        Inheriting class must take care of implementing this method
        :return: list of test files
        """
        raise NotImplementedError

    def get_test_labels(self):
        """
        Inheriting class must take care of implementing this method
        :return: list of labels in sync with test files
        """
        raise NotImplementedError

    def get_predefined_labels(self):
        """
        Inheriting class must take care of implementing this method
        :return: list of predefined lables. Eg: ["_silence_", "_background_noise_"]
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Inheriting class must take care of implementing this method
        :return: List of lables for the dataset under consideration
        """
        raise NotImplementedError

    def _setup_labels(self):
        """
        Sets up the label indexer.
        Note: This needs to be called while the inheriting class is getting initialized
        :return: None
        """
        self._labels = self.get_labels()
        self._labels = self.get_predefined_labels() + list(self._labels)
        self._labels = sorted(self._labels)

        self._labels_2_index = {label.lower():i for i, label in enumerate([self._unknown_label] + self._labels)}
        self._index_2_labels = {i: label for label, i in self._labels_2_index.items()}

        self._labels_dim = len(self._labels_2_index)
        return None


    def get_label_2_index(self, label):
        """
        Returns the index of the label, considering the predefined labels
        :param label: string
        :return: index: int
        """
        return self._labels_2_index.get(label, 0) #return unknown index when not found

    def get_index_2_label(self, index):
        """
        Returns the label string, considering the predefined labels
        :param index: int
        :return: label: string
        """
        return self._index_2_labels.get(index,  self._unknown_label)

    def get_one_hot_encoded(self, label):
        """
        Returns the one-hot encoded array of the label
        :param label: string
        :return: np.array
        """
        label = str(label, 'utf-8').lower()
        vector = np.zeros(self._labels_dim, dtype=int)
        index = self.get_label_2_index(label=label)
        vector[index] = 1
        return vector

    def store_labels_index_map(self, file_name="labels_index_map.json"):
        """
        Stores teh current label index as json, as per the path
        `labels_index_map_store_path` specified in the params
        Full store path: labels_index_map_store_path/dataset_name/
        :return: None
        """
        directory = os.path.join(self._hparams["labels_index_map_store_path"],
                        self.get_dataset_name())

        if not os.path.isdir(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, file_name), 'w') as file:
            json.dump(self._labels_2_index, file)

    def load_labels_index_map(self, file_path):
        """
        Reads the JSON file from the path and loads them into dataset label indexer
        :param file_path: File path of the JSON
        :return: None
        """
        with open(file_path) as handle:
            self._labels_2_index = json.loads(handle.read())
            self._index_2_labels = {i: label.lower() for label, i in self._labels_2_index.items()}
            self._labels_dim = len(self._labels_2_index)
