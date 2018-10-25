import sys

from shabda.sr_config.sr_config import *

sys.path.append("../")

import tensorflow as tf
import numpy as np
from shabda.dataset.iterators.data_iterator import DataIterator
from shabda.dataset.feature_types import MFCCFeature
from tqdm import tqdm

import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from six.moves import xrange # pylint: disable=redefined-builtin

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

from shabda.helpers.print_helper import *
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

from scipy.io import wavfile


class AudioMFCC(DataIterator):
    def __init__(self, tf_sess, batch_size, num_epochs, audio_preprocessor):
        DataIterator.__init__(self)

        self._tf_sess = tf_sess

        self._feature_type = MFCCFeature

        self._audio_preprocessor = audio_preprocessor
        self._batch_size = batch_size
        self._num_epochs = num_epochs

        self.background_data = self._audio_preprocessor.background_data
        self.word_to_index = self._audio_preprocessor.word_to_index

    def melspectrogram(self, sample_rate, audio):

        # sample_rate, audio = wavfile.read(wave_file_path)
        # S = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=64)
        # print_info(S.shape)
        # log_S = librosa.power_to_db(S, ref=np.max)

        # mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=98, n_fft=1024, hop_length=406)
        mfcc = mfcc.astype(np.float32)
        # print_info(mfcc.shape)
        # print(mfcc)
        # delta_mfcc = librosa.feature.delta(mfcc)
        return mfcc.flatten()

    def data_generator(self, data, params, mode='train'):
        def generator():
            if mode == 'train':
                np.random.shuffle(data)
            # Feel free to add any augmentation
            for i, label_file_dict in tqdm(enumerate(data), desc=mode):
                fname = label_file_dict["file"]
                label = label_file_dict["label"]
                print_error(str(i) + " ======> " + fname)
                label_id = self.word_to_index[label]
                try:
                    sample_rate, wav = wavfile.read(fname)
                    # wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                    L = 16000  # be aware, some files are shorter than 1 sec!
                    if len(wav) < L:
                        continue

                    yield {self._feature_type.FEATURE_1 : self.melspectrogram(sample_rate, wav),
                     self._feature_type.TARGET: np.int32(label_id)}
                except Exception as err:
                    print(err, label_id, fname)

        return generator

    def get_train_input_fn(self):
        train_input_fn = generator_input_fn(
            x=self.data_generator(self._audio_preprocessor.get_train_data(), None, 'train'),
            target_key='target',  # you could leave target_key in features, so labels in model_handler will be empty
            batch_size=self._batch_size,
            shuffle=True,
            num_epochs=self._num_epochs,
            queue_capacity=3 * self._batch_size + 10,
            num_threads=1,
        )

        return train_input_fn

    def get_val_input_fn(self):
        val_input_fn = generator_input_fn(
            x=self.data_generator(self._audio_preprocessor.get_val_data(), None, 'val'),
            target_key='target',
            batch_size=self._batch_size,
            shuffle=True,
            num_epochs=1,
            queue_capacity=3 * self._batch_size + 10,
            num_threads=1,
        )

        return val_input_fn
