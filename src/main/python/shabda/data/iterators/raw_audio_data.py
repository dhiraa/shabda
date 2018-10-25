import sys

from speech_recognition.sr_config.sr_config import *

sys.path.append("../")

from speech_recognition.dataset.iterators.data_iterator import DataIterator
from speech_recognition.dataset.feature_types import RawWavAudio
from tqdm import tqdm

import numpy as np


from sarvam.helpers.print_helper import *
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

from scipy.io import wavfile

class AudioMFCC(DataIterator):
    def __init__(self, tf_sess, batch_size, num_epochs, audio_preprocessor):
        DataIterator.__init__(self)

        self._tf_sess = tf_sess

        self._feature_type = RawWavAudio

        self._audio_preprocessor = audio_preprocessor
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        # self._audio_sampling_settings = audio_sampling_settings

    def data_generator(self, data, params, mode='train'):
        def generator():
            if mode == 'train':
                np.random.shuffle(data)
            # Feel free to add any augmentation
            for i, label_file_dict in tqdm(enumerate(data), desc=mode):
                fname = label_file_dict["file"]
                label = label_file_dict["label"]
                label_id = self._audio_preprocessor.word_to_index[label]
                try:
                    sample_rate, wav = wavfile.read(fname)
                    wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                    L = 16000  # be aware, some files are shorter than 1 sec!
                    if len(wav) < L:
                        continue

                    # let's generate more silence!
                    samples_per_file = 1 if label_id != self._audio_preprocessor.word_to_index['_silence_'] else 20
                    for _ in range(samples_per_file):
                        if len(wav) > L:
                            beg = np.random.randint(0, len(wav) - L)
                        else:
                            beg = 0
                    wav = wav[beg: beg + L]

                    # print_error(i)
                    # print(fname)
                    # print_info(wav)
                    # print_debug(wav.sum())

                    yield {self._feature_type.FEATURE_1 : wav,
                     self._feature_type.TARGET: np.int32(label_id)}

                except Exception as err:
                    print_error(str(err) + " " + str(label_id) + " " + fname)

        return generator

    def get_train_input_fn(self):
        train_input_fn = generator_input_fn(
            x=self.data_generator(self._audio_preprocessor.get_train_files(), None, 'train'),
            target_key=self._feature_type.TARGET,  # you could leave target_key in features, so labels in model_handler will be empty
            batch_size=self._batch_size,
            shuffle=True,
            num_epochs=self._num_epochs,
            queue_capacity=3 * self._batch_size + 10,
            num_threads=1,
        )

        return train_input_fn

    def get_val_input_fn(self):
        val_input_fn = generator_input_fn(
            x=self.data_generator(self._audio_preprocessor.get_val_files(), None, 'val'),
            target_key=self._feature_type.TARGET,
            batch_size=self._batch_size,
            shuffle=True,
            num_epochs=1,
            queue_capacity=3 * self._batch_size + 10,
            num_threads=1,
        )

        return val_input_fn
