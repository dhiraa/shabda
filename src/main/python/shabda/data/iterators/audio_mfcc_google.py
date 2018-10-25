import sys

from speech_recognition.sr_config.sr_config import *

sys.path.append("../")

import tensorflow as tf
import numpy as np
from speech_recognition.dataset.iterators.data_iterator import DataIterator
from speech_recognition.dataset.feature_types import MFCCFeature
from nlp.text_classification.tc_utils.tf_hooks.data_initializers import IteratorInitializerHook
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

from sarvam.helpers.print_helper import *
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

        self._audio_sampling_settings = self._feature_type.audio_sampling_settings

        desired_samples = self._audio_sampling_settings['desired_samples']

        self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])
        self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
        self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
        self.background_data_placeholder_ = tf.placeholder(tf.float32, [desired_samples, 1])
        self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])

        self.background_data = self._audio_preprocessor.background_data
        self.word_to_index = self._audio_preprocessor.word_to_index

        self.prepare_processing_graph(self._audio_sampling_settings)


    def prepare_processing_graph(self, model_settings):
        """
        Builds a TensorFlow graph to apply the input distortions.
        Creates a graph that loads a WAVE file, decodes it, scales the volume,
        shifts it in time, adds in background noise, calculates a spectrogram, and
        then builds an MFCC fingerprint from that.
        This must be called with an active TensorFlow session running, and it
        creates multiple placeholder inputs, and one output:
          - wav_filename_placeholder_: Filename of the WAV to load.
          - foreground_volume_placeholder_: How loud the main clip should be.
          - time_shift_padding_placeholder_: Where to pad the clip.
          - time_shift_offset_placeholder_: How much to move the clip in time.
          - background_data_placeholder_: PCM sample data for background noise.
          - background_volume_placeholder_: Loudness of mixed-in background.
          - mfcc_: Output 2D fingerprint of processed audio.
        :param model_settings: Information about the current model being trained.
        :return: 
        """

        desired_samples = model_settings['desired_samples']

        wav_loader = io_ops.read_file(self.wav_filename_placeholder_)

        wav_decoder = contrib_audio.decode_wav(
            wav_loader, desired_channels=1, desired_samples=desired_samples)
        # Allow the audio sample's volume to be adjusted.
        self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
        scaled_foreground = tf.multiply(wav_decoder.audio,
                                        self.foreground_volume_placeholder_)
        # Shift the sample's start position, and pad any gaps with zeros.

        padded_foreground = tf.pad(
            scaled_foreground,
            self.time_shift_padding_placeholder_,
            mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground,
                                     self.time_shift_offset_placeholder_,
                                     [desired_samples, -1])
        # Mix in background noise.

        background_mul = tf.multiply(self.background_data_placeholder_,
                                     self.background_volume_placeholder_)
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
        # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
        spectrogram = contrib_audio.audio_spectrogram(
            background_clamp,
            window_size=model_settings['window_size_samples'],
            stride=model_settings['window_stride_samples'],
            magnitude_squared=True)
        self.mfcc_ = contrib_audio.mfcc(
            spectrogram,
            wav_decoder.sample_rate,
            dct_coefficient_count=model_settings['dct_coefficient_count'])


    def get_data(self,
                 candidates,
                 how_many,
                 offset,
                 audio_sampling_settings,
                 background_frequency,
                 background_volume_range,
                 time_shift,
                 mode,
                 sess):
        """
        Gather samples from the data set, applying transformations as needed.
        When the mode is 'training', a random selection of samples will be returned,
        otherwise the first N clips in the partition will be used. This ensures that
        validation always uses the same samples, reducing noise in the metrics.
        :param how_many: Desired number of samples to return. -1 means the entire
            contents of this partition. 
        :param offset: Where to start when fetching deterministically.
          model_settings: Information about the current model being trained.
        :param audio_sampling_settings: 
        :param background_frequency: How many clips will have background noise, 0.0 to
            1.0.
        :param background_volume_range: How loud the background noise will be.
        :param time_shift: How much to randomly shift the clips by in time.
        :param mode: Which partition to use, must be 'training', 'validation', or
            'testing'.
        :param sess: TensorFlow session that was active when processor was created.
        :return: List of sample data for the transformed samples, and list of label indexes
        """
        # Pick one of the partitions to choose samples from.
        def generator():
            if how_many == -1:
                sample_count = len(candidates)
            else:
                sample_count = max(0, min(how_many, len(candidates) - offset))

            # Data and labels will be populated and returned.
            data = np.zeros((sample_count, audio_sampling_settings['fingerprint_size']))
            labels = np.zeros(sample_count)
            desired_samples = audio_sampling_settings['desired_samples']

            use_background = self.background_data and (mode == 'training')
            pick_deterministically = (mode != 'training')

            # Use the processing graph we created earlier to repeatedly to generate the
            # final output sample data we'll use in training.
            for i in tqdm(xrange(offset, offset + sample_count)):
                # Pick which audio sample to use.
                if how_many == -1 or pick_deterministically:
                    sample_index = i
                else:
                    sample_index = np.random.randint(len(candidates))
                sample = candidates[sample_index]

                # If we're time shifting, set up the offset for this sample.
                if time_shift > 0:
                    time_shift_amount = np.random.randint(-time_shift, time_shift)
                else:
                    time_shift_amount = 0

                if time_shift_amount > 0:
                    time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                    time_shift_offset = [0, 0]
                else:
                    time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                    time_shift_offset = [-time_shift_amount, 0]

                input_dict = {
                    self.wav_filename_placeholder_: sample['file'],
                    self.time_shift_padding_placeholder_: time_shift_padding,
                    self.time_shift_offset_placeholder_: time_shift_offset,
                }

                # Choose a section of background noise to mix in.
                if use_background:
                    background_index = np.random.randint(len(self.background_data))
                    background_samples = self.background_data[background_index]
                    background_offset = np.random.randint(
                        0, len(background_samples) - audio_sampling_settings['desired_samples'])
                    background_clipped = background_samples[background_offset:(background_offset + desired_samples)]
                    background_reshaped = background_clipped.reshape([desired_samples, 1])
                    if np.random.uniform(0, 1) < background_frequency:
                        background_volume = np.random.uniform(0, background_volume_range)
                    else:
                        background_volume = 0
                else:
                    background_reshaped = np.zeros([desired_samples, 1])
                    background_volume = 0

                input_dict[self.background_data_placeholder_] = background_reshaped
                input_dict[self.background_volume_placeholder_] = background_volume

                # If we want silence, mute out the main sample but leave the background.
                if sample['label'] == SILENCE_LABEL:
                    input_dict[self.foreground_volume_placeholder_] = 0
                else:
                    input_dict[self.foreground_volume_placeholder_] = 1

                    mfcc_data = sess.run(self.mfcc_, feed_dict=input_dict).flatten()
                    label_index = self.word_to_index[sample['label']]
                    # print_error(str(i) + " ======> " + sample['file'])
                    # print_info(mfcc_data)
                    # print_info(label_index)
                    # yield dict(wav=mfcc_data, target=np.array(label_index))
                    yield {self._feature_type.FEATURE_1: mfcc_data,
                           self._feature_type.TARGET: np.array(label_index)}
        return generator

    def get_train_input_fn(self):
        train_input_fn = generator_input_fn(
            x=self.get_data(candidates=self._audio_preprocessor.get_train_files(),
                            how_many=-1,
                            offset=0,
                            audio_sampling_settings=self._audio_sampling_settings,
                            background_frequency=BACKGROUND_FREQUENCY,
                            background_volume_range=BACKGROUND_VOLUME,
                            time_shift=TIME_SHIFT_MS,
                            mode="training",
                            sess=self._tf_sess),
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
            x=self.get_data(candidates=self._audio_preprocessor.get_val_files(),
                            how_many=-1,
                            offset=0,
                            audio_sampling_settings=self._audio_sampling_settings,
                            background_frequency=BACKGROUND_FREQUENCY,
                            background_volume_range=BACKGROUND_VOLUME,
                            time_shift=TIME_SHIFT_MS,
                            mode="validation",
                            sess=self._tf_sess),
            target_key=self._feature_type.TARGET,
            batch_size=self._batch_size,
            shuffle=True,
            num_epochs=1,
            queue_capacity=3 * self._batch_size + 10,
            num_threads=1,
        )

        return val_input_fn