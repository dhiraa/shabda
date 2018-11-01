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
FreeSound dataset
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pandas as pd
from sklearn.model_selection import train_test_split
from overrides import overrides

from shabda.data.dataset.internal.audio_dataset_base import AudioDatasetBase
from shabda.helpers.print_helper import *


class FreeSoundAudioDataset(AudioDatasetBase):
    """
    Dataset class that encapsulates the data collected form @ https://www.kaggle.com/c/freesound-audio-tagging

    Basically the dataset is collection of audio files and corresponding labels (41 tags to be specific).
    The predicted_class and test data file names and tags are given as .csv file as predicted_class.csv and test.csv.
    We use these info and files location to load the data for our purpose.

    Data can be downloaded to the `data` dir using the Kaggle API. Refer `data` dir for more info.
    """
    def __init__(self, hparams=None):
        # TODO: test "super(AudioDatasetBase, self).__init__(self, hparams=hparams)"
        AudioDatasetBase.__init__(self, hparams=hparams)

        self.train_csv_path = self._hparams.train_csv_path
        self.val_csv_path = self._hparams.val_csv_path
        self.test_csv_path = self._hparams.test_csv_path

        self._train_audio_files_dir = self._hparams.train_audio_files_dir
        self._val_audio_files_dir = self._hparams.val_audio_files_dir
        self._test_audio_files_dir = self._hparams.test_audio_files_dir

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.init()

    @overrides
    def init(self):
        self._load_data_info()
        self._setup_labels()

    @staticmethod
    def default_hparams():
        freesound_dataset = {
            "labels_index_map_store_path": "/tmp/shabda/",
            "n_classes": 41,
            "train_csv_path" : "data/freesound-audio-tagging/input/train.csv",
            "val_csv_path" : None,  # we dont have any validation csv file as such
            "test_csv_path" : "./data/freesound-audio-tagging/input/sample_submission.csv",
            "train_audio_files_dir" : "./data/freesound-audio-tagging/input/audio_train/",
            "val_audio_files_dir" : "./data/freesound-audio-tagging/input/audio_train/",
            "test_audio_files_dir" : "./data/freesound-audio-tagging/input/audio_test/"
        }
        return freesound_dataset

    @overrides
    def get_dataset_name(self):
        return "free_sound_dataset"

    def _load_data_info(self):
        # fname label manually_verified
        print_info("Train data info DF : " + self.train_csv_path)
        print_info("Test data info DF : " +  self.test_csv_path)

        # We use train_test_split API to split predicted_class set into tran and val set
        self.train_df = pd.read_csv(self.train_csv_path)
        if self.val_csv_path == None:
            self.train_df, self.val_df = train_test_split(self.train_df, test_size=0.2)
        else:
            self.val_df = pd.read_csv(self.val_csv_path)

        self.test_df = pd.read_csv(self.test_csv_path)

    @overrides
    def get_num_train_samples(self):
        return self.train_df.shape[0]

    @overrides
    def get_labels(self):
        return self.train_df["label"].unique()

    @overrides
    def get_predefined_labels(self):
        """
        No predefined labels
        :return: list: empty
        """
        return []

    @overrides
    def get_train_files(self):
        """
        Returns the list of train file paths from Freesound dataset
        :return:
        """
        files = []
        for fname in  self.train_df['fname']:
            files.append(self._hparams.train_audio_files_dir + "/" + fname)
        return files

    @overrides
    def get_val_files(self):
        """
        Returns the list of validation file paths from Freesound dataset
        :return:
        """
        files = []
        for fname in self.val_df['fname']:
            files.append(self._hparams.val_audio_files_dir + "/" + fname)
        return files

    @overrides
    def get_train_labels(self):
        """
        Returns the list of train labels from Freesound dataset
        :return: list of one-hot encoded labels
        """
        return self.train_df["label"]

    @overrides
    def get_val_labels(self):
        """
        Returns the list of validation labels from Freesound dataset
        :return: list of one-hot encoded labels
        """
        return self.val_df["label"]
