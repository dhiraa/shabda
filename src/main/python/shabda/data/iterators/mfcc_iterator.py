import tensorflow as tf
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
from tensorflow.contrib import ffmpeg

from tqdm import tqdm
import numpy as np
from shabda.data.dataset.freesound_dataset import FreeSoundAudioDataset
from overrides import overrides
from shabda.data.iterators.internal.free_sound_data_iterator_base import FreeSoundDataIteratorBase
from shabda.helpers import audio
from shabda.helpers.print_helper import *
#
# class IteratorInitializerHook(tf.train.SessionRunHook):
#     """Hook to initialise audio_utils iterator after Session is created."""
#
#     def __init__(self):
#         super(IteratorInitializerHook, self).__init__()
#         self.iterator_initializer_func = None
#
#     def after_create_session(self, session, coord):
#         """Initialise the iterator after the session has been created."""
#         self.iterator_initializer_func(session)
        
class MFCCDataIterator(FreeSoundDataIteratorBase):
    def __init__(self, hparams, dataset: FreeSoundAudioDataset):
        super(MFCCDataIterator, self).__init__(hparams, dataset)

        if not isinstance(dataset, FreeSoundAudioDataset):
            raise AssertionError("dataset should be FreeSoundAudioDataset")


    def get_mfcc_2d_feature(self, file_path, label):
        data, sample_rate = audio.load_wav_audio_file(file_path=file_path)

        data = audio.pad_data_array(data=data,
                                    max_audio_length=self._max_audio_length)

        data = audio.get_frequency_spectrum(data=data, n_mfcc=self._n_mfcc,
                                            sampling_rate=self._sampling_rate)
        #
        data = np.expand_dims(data, axis=-1) #to make it compatible with CNN network
        data = data.flatten(order="C") #row major

        label = self._dataset.get_label_2_index(label)
        return data, label

    def _get_train_input_fn(self):
        # print(self._train_files_path, self._train_labels)
        dataset = tf.data.Dataset.from_tensor_slices((self._train_files_path, self._train_labels))
        dataset = dataset.map(
            lambda filename, label: tuple(tf.py_func(
                self.get_mfcc_2d_feature, [filename, label], [tf.double, tf.int32])),
            num_parallel_calls=4)

        dataset = dataset.shuffle(len(self._train_labels))
        # dataset = dataset.map(self.get_mfcc_2d_feature, num_parallel_calls=4)
        dataset = dataset.repeat(self._num_epochs)
        dataset = dataset.batch(self._batch_size)

        return dataset

        def _input_fn():

            iterator = dataset.make_one_shot_iterator()

            batch = iterator.get_next()

            features, labels = {}, {}
            for key, value in batch.items():
                if key.startswith('source_'):
                    features[key] = value
                else:
                    labels[key] = value
            return features, labels

        return _input_fn


    def _get_val_input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((self._val_files_path, self._val_labels))
        dataset = dataset.map(
            lambda filename, label: tuple(tf.py_func(
                self.get_mfcc_2d_feature, [filename, label], [tf.double, label.dtype])),
            num_parallel_calls=4)
        dataset = dataset.shuffle(len(self._val_labels))
        # dataset = dataset.map(self.get_mfcc_2d_feature, num_parallel_calls=4)

        dataset = dataset.repeat(self._num_epochs)
        dataset = dataset.batch(self._batch_size)

        return dataset

        def _input_fn():

            iterator = dataset.make_one_shot_iterator()

            batch = iterator.get_next()

            features, labels = {}, {}
            for key, value in batch.items():
                if key.startswith('source_'):
                    features[key] = value
                else:
                    labels[key] = value
            return features, labels

        return _input_fn

    def _get_test_input_function(self):
        raise NotImplementedError


    # def data_generator(self, wav_files_path, labels, batch_size, mode):
    #
    #     if labels is not None:
    #         assert(len(wav_files_path) == len(labels))
    #
    #     number_of_examples = (len(wav_files_path) // batch_size) * batch_size
    #
    #     def generator():
    #         for i in tqdm(range(number_of_examples), desc=mode):
    #             data, sample_rate = audio.load_wav_audio_file(file_path=wav_files_path[i])
    #
    #             data = audio.pad_data_array(data=data,
    #                                         max_audio_length=self._max_audio_length)
    #
    #             #TODO Commented this code
    #             data = audio.get_frequency_spectrum(data=data, n_mfcc=self._n_mfcc,
    #                                                 sampling_rate=self._sampling_rate)
    #             #
    #             data = np.expand_dims(data, axis=-1) #to make it compatible with CNN network
    #
    #             #TODO added this line
    #             # data = FreeSoundDataset.audio_norm(data)[:, np.newaxis]
    #
    #             data = data.flatten(order="C") #row major
    #
    #             if labels is not None:
    #                 res = {"features": data,"labels":labels[i]}
    #             else:
    #                 res = {"features": data,"labels": np.array()}
    #
    #             yield res
    #
    #     return generator
    #
    # @overrides
    # def _get_train_input_fn(self):
    #     train_wav_files_path = self.train_wav_files_path
    #     labels = self.train_labels
    #
    #     train_input_fn = generator_input_fn(
    #         x=self.data_generator(train_wav_files_path, labels, self._batch_size, 'train'),
    #         target_key=None,  # you could leave target_key in features, so labels in model_handler will be empty
    #         batch_size=self._batch_size,
    #         shuffle=True,
    #         num_epochs=1,
    #         queue_capacity=2000,
    #         num_threads=1,
    #     )
    #
    #     return train_input_fn
    #
    # @overrides
    # def _get_val_input_fn(self):
    #
    #     train_wav_files_path = self.val_wav_files_path
    #     labels = self.val_labels
    #
    #     val_input_fn = generator_input_fn(
    #         x=self.data_generator(train_wav_files_path, labels, self._batch_size, 'val'),
    #         target_key=None,
    #         batch_size=self._batch_size,
    #         shuffle=True,
    #         num_epochs=1,
    #         queue_capacity=2000,
    #         num_threads=1,
    #     )
    #
    #     return val_input_fn
    #
    # @overrides
    # def _get_test_input_function(self):
    #
    #     test_wav_files_path = self.test_wav_files_path
    #     val_input_fn = generator_input_fn(
    #         x=self.data_generator(test_wav_files_path, None, 1, 'test'),
    #         target_key=None,
    #         batch_size=1,
    #         shuffle=False,
    #         num_epochs=1,
    #         queue_capacity=2000,
    #         num_threads=1,
    #     )
    #
    #     return val_input_fn
