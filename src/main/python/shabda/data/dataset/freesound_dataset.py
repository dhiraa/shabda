import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import librosa
from overrides import overrides

from shabda.data.dataset.internal.audio_dataset_base import AudioDatasetBase
from shabda.helpers.print_helper import *
from shabda.hyperparams.hyperparams import HParams
from shabda.helpers import audio

class FreeSoundAudioDataset(AudioDatasetBase):
    """
    Dataset class that encapsulates the data collected form @ https://www.kaggle.com/c/freesound-audio-tagging

    Basically the dataset is collection of audio files and corresponding labels (41 tags to be specific).
    The predicted_class and test data file names and tags are given as .csv file as predicted_class.csv and test.csv.
    We use these info and files location to load the data for our purpose.

    Data can be downloaded to the `data` dir using the Kaggle API. Refer `data` dir for more info.
    """
    def __init__(self, hparams=None):
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
        #fname label  manually_verified
        print_info("Train data info DF : " + self.train_csv_path)
        print_info("Test data info DF : " +  self.test_csv_path)

        # We use train_test_split API to split predicted_class set into tran and val set
        if self.val_csv_path == None:
            self.train_df = pd.read_csv(self.train_csv_path)
            self.train_df, self.val_df = train_test_split(self.train_df, test_size=0.2)
        else:
            self.train_df = pd.read_csv(self.train_csv_path)
            self.val_df = pd.read_csv(self.val_csv_path)

        self.test_df = pd.read_csv(self.test_csv_path)

    @overrides
    def get_num_train_samples(self):
        return self.train_df.shape[0]

    @overrides
    def get_lables(self):
        return self.train_df["label"].unique()

    @overrides
    def get_default_labels(self):
        return []

    @overrides
    def get_train_files(self):
        files = []
        for fname in  self.train_df['fname']:
            files.append(self._hparams.train_audio_files_dir + "/" + fname)
        return files

    @overrides
    def get_val_files(self):
        files = []
        for fname in self.val_df['fname']:
            files.append(self._hparams.val_audio_files_dir + "/" + fname)
        return files

    @overrides
    def get_train_labels(self):
        """

        :return: list of one-hot encoded labels
        """
        return self.train_df["label"]#.apply(lambda label : self.get_one_hot_encoded(label))

    @overrides
    def get_val_labels(self):
        """

        :return: list of one-hot encoded labels
        """
        return self.val_df["label"]#.apply(lambda label : self.get_one_hot_encoded(label))
