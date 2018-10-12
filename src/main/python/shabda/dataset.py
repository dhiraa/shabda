import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import librosa

from shabda.helpers.print_helper import *

class FreeSoundDataset():
    def __init__(self,
                 dev_csv_path="data/freesound-audio-tagging/input/train.csv",
                 val_csv_path=None,
                 test_csv_path="data/freesound-audio-tagging/input/sample_submission.csv",
                 dev_audio_files_dir="data/freesound-audio-tagging/input/audio_train/",
                 val_audio_files_dir="data/freesound-audio-tagging/input/audio_train/",
                 test_audio_files_dir="data/freesound-audio-tagging/input/audio_test/"):

        self.dev_csv_path = dev_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path

        self._dev_audio_files_dir = dev_audio_files_dir
        self._val_audio_files_dir = val_audio_files_dir
        self._test_audio_files_dir = test_audio_files_dir

        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load_data_info(self):
        #fname label  manually_verified
        print_info("Train data info DF : " + self.dev_csv_path)
        print_info("Test data info DF : " +  self.test_csv_path)

        if self.val_csv_path == None:
            self.train_df = pd.read_csv(self.dev_csv_path)
            self.train_df, self.val_df = train_test_split(self.train_df, test_size=0.2)
        else:
            self.train_df = pd.read_csv(self.dev_csv_path)
            self.val_df = pd.read_csv(self.val_csv_path)

        # self.train_df = pd.concat([self.train_df, pd.get_dummies(self.train_df["label"])], axis=1)
        # self.val_df = pd.concat([self.val_df, pd.get_dummies(self.val_df["label"])], axis=1)

        self.test_df = pd.read_csv(self.test_csv_path)

        print_error(self.train_df.head(10))

    def get_num_samples(self):
        return self.train_df.shape[0]

    def get_dev_wav_files(self):
        return self.train_df['fname']

    def get_val_wav_files(self):
        return self.val_df['fname']

    def get_dev_labels_df(self):
        return pd.get_dummies(self.train_df["label"])

    def get_val_labels_df(self):
        return pd.get_dummies(self.val_df["label"])


    @property
    def dev_audio_files_dir(self):
        return self._dev_audio_files_dir

    @property
    def val_audio_files_dir(self):
        return self._val_audio_files_dir

    @property
    def test_audio_files_dir(self):
        return self._test_audio_files_dir

    @staticmethod
    def pad_data_array(data, max_audio_length):
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


    @staticmethod
    def get_frequency_spectrum(data, sampling_rate, n_mfcc):
        return librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc, n_fft=1024, hop_length=345) #with sr=44KHz and hop_length=345 to get n_mfcc x 256

    @staticmethod
    def load_wav_audio_file(file_path):
        data, sample_rate = librosa.core.load(file_path, sr=None, res_type="kaiser_fast")
        # data, sample_rate = librosa.core.load("data/freesound-audio-tagging/input/audio_train/00044347.wav", res_type="kaiser_fast")
        return data, sample_rate

        
