import numpy as np
from shabda.data.dataset.freesound_dataset import FreeSoundAudioDataset
from overrides import overrides
from shabda.data.iterators.internal.free_sound_data_iterator_base import FreeSoundDataIteratorBase
from shabda.helpers import audio
from shabda.helpers.print_helper import *

        
class MFCCDataIterator(FreeSoundDataIteratorBase):
    def __init__(self, hparams, dataset: FreeSoundAudioDataset):
        super(MFCCDataIterator, self).__init__(hparams, dataset)

        if not isinstance(dataset, FreeSoundAudioDataset):
            raise AssertionError("dataset should be FreeSoundAudioDataset")


    @overrides
    def _user_map_func(self, file_path, label):
        data, sample_rate = audio.load_wav_audio_file(file_path=file_path)

        data = audio.pad_data_array(data=data,
                                    max_audio_length=self._max_audio_length)

        data = audio.get_frequency_spectrum(data=data, n_mfcc=self._n_mfcc,
                                            sampling_rate=self._sampling_rate)

        data = np.expand_dims(data, axis=-1) #to make it compatible with CNN network
        data = data.flatten(order="C") #row major

        label = self._dataset.get_one_hot_encoded(label)
        return data, label

    @overrides
    def _user_resize_func(self, data, label):
        # data.set_shape([None,None,None])
        # label.set_shape([None, None])
        # data = tf.image.resize_images(data, [128,33])
        # data = tf.reshape(data, shape=[128,33])
        # label = tf.reshape(label, shape=[42])
        return data, label