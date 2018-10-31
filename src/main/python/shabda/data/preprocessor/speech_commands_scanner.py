# import hashlib
# import math
# import os
# import random
# import re
#
# import tensorflow as tf
# from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# from tensorflow.python.ops import io_ops
# from tensorflow.python.platform import gfile
# from tensorflow.python.util import compat
#
# from shabda.helpers.audio import prepare_words_list
# from shabda.helpers.print_helper import *
# from shabda.helpers.downloaders import maybe_download_and_extract_dataset
# from shabda.data.preprocessor.preprocessor_interface import IPreprocessor
# from shabda.hyperparams.hyperparams import HParams
#
#
# # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/input_data.py
#
# def which_set(filename, validation_percentage, testing_percentage):
#     """Determines which data partition the file should belong to.
#     We want to keep files in the same training, validation, or testing sets even
#     if new ones are added over time. This makes it less likely that testing
#     samples will accidentally be reused in training when long runs are restarted
#     for example. To keep this stability, a hash of the filename is taken and used
#     to determine which set it should belong to. This determination only depends on
#     the name and the set proportions, so it won't change as other files are added.
#     It's also useful to associate particular files as related (for example words
#     spoken by the same person), so anything after '_nohash_' in a filename is
#     ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
#     'bobby_nohash_1.wav' are always in the same set, for example.
#     Args:
#       filename: File path of the data sample.
#       validation_percentage: How much of the data set to use for validation.
#       testing_percentage: How much of the data set to use for testing.
#     Returns:
#       String, one of 'training', 'validation', or 'testing'.
#     """
#
#     MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1
#     base_name = os.path.basename(filename)
#
#     # We want to ignore anything after '_nohash_' in the file name when
#     # deciding which set to put a wav in, so the data set creator has a way of
#     # grouping wavs that are close variations of each other.
#     hash_name = re.sub(r'_nohash_.*$', '', base_name)
#
#     # This looks a bit magical, but we need to decide whether this file should
#     # go into the training, testing, or validation sets, and we want to keep
#     # existing files in the same set even if more files are subsequently
#     # added.
#     # To do that, we need a stable way of deciding based on just the file name
#     # itself, so we do a hash of that and then use that to generate a
#     # probability value that we use to assign it.
#     hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
#     percentage_hash = ((int(hash_name_hashed, 16) %
#                         (MAX_NUM_WAVS_PER_CLASS + 1)) *
#                        (100.0 / MAX_NUM_WAVS_PER_CLASS))
#
#     if percentage_hash < validation_percentage:
#         result = 'validation'
#     elif percentage_hash < (testing_percentage + validation_percentage):
#         result = 'testing'
#     else:
#         result = 'training'
#
#     return result
#
# class SpeechCommandsDirectoryProcessor(IPreprocessor):
#     """Handles loading, partitioning, and preparing audio training data."""
#
#     def __init__(self, hparams=None):
#
#         # IPreprocessor.__init__()
#
#         self._hparams = HParams(hparams=hparams, default_hparams=None)
#
#         self._config = self._hparams.speech_recognition_dataest
#
#         data_url = self._config.data_url
#         data_dir = self._config.data_dir
#         silence_percentage = self._config.silence_percentage
#         unknown_percentage = self._config.unknown_percentage
#         possible_commands = self._config.possible_commands
#         validation_percentage = self._config.validation_percentage
#         testing_percentage = self._config.testing_percentage
#         random_seed = self._config.random_seed
#
#         maybe_download_and_extract_dataset(data_url, data_dir)
#
#         self.data_dir = data_dir
#
#         # possible_commands = prepare_words_list(possible_commands)
#
#         random.seed(random_seed)
#
#         self.data_buckets = {'validation': [], 'testing': [], 'training': []}
#         self.word_to_index = {}
#
#         self.POSSIBLE_SPEECH_COMMANDS = possible_commands
#
#         self.prepare_data_index(silence_percentage,
#                                 unknown_percentage,
#                                 self.POSSIBLE_SPEECH_COMMANDS,
#                                 validation_percentage,
#                                 testing_percentage)
#
#         self.background_data = []
#         self.prepare_background_data()
#
#         self.NUM_SAMPLES = len(self.data_buckets["training"])
#
#     def prepare_data_index(self,
#                            silence_percentage,
#                            unknown_percentage,
#                            wanted_words,
#                            validation_percentage,
#                            testing_percentage):
#         """
#         Prepares a list of the samples organized by set and label.
#         The training loop needs a list of all the available data, organized by
#         which partition it should belong to, and with ground truth labels attached.
#         This function analyzes the folders below the `data_dir`, figures out the
#         right labels for each file based on the name of the subdirectory it belongs to,
#         and uses a stable hash to assign it to a data set partition.
#         :param silence_percentage: How much of the resulting data should be background.
#         :param unknown_percentage: How much should be audio outside the wanted classes.
#         :param wanted_words: Labels of the classes we want to be able to recognize.
#         :param validation_percentage: How much of the data set to use for validation.
#         :param testing_percentage: How much of the data set to use for testing.
#         :return:  Dictionary containing a list of file information for each set partition,
#           and a lookup map for each class to determine its numeric index.
#         :raises Exception: If expected files are not found.
#         """
#
#         # Make sure the shuffling and picking of unknowns is deterministic.
#
#         wanted_words_index = {}
#
#         for index, wanted_word in enumerate(wanted_words):
#             wanted_words_index[
#                 wanted_word] = index + 2  # 0 and 1 goes for SILENCE_LABEL, UNKNOWN_WORD_LABEL respectively
#
#         unknown_buckets = {'validation': [], 'testing': [], 'training': []}
#         all_words = {}
#
#         # Look through all the subfolders to find audio samples
#         search_path = os.path.join(self.data_dir, '*', '*.wav')
#
#         for wav_path in gfile.Glob(search_path):
#             _, word = os.path.split(os.path.dirname(wav_path))
#             word = word.lower()
#             # Treat the '_background_noise_' folder as a special case, since we expect
#             # it to contain long audio samples we mix in to improve training.
#             if word == self._config.background_noise_dir_name:
#                 continue
#             all_words[word] = True
#             set_index = which_set(wav_path, validation_percentage, testing_percentage)
#             # If it's a known class, store its detail, otherwise add it to the list
#             # we'll use to train the unknown label.
#             if word in wanted_words_index:
#                 self.data_buckets[set_index].append({'label': word, 'file': wav_path})
#             else:
#                 unknown_buckets[set_index].append({'label': word, 'file': wav_path})
#
#         if not all_words:
#             raise Exception('No .wavs found at ' + search_path)
#
#         for index, wanted_word in enumerate(wanted_words):
#             if wanted_word not in all_words:
#                 raise Exception('Expected to find ' + wanted_word +
#                                 ' in labels but only found ' +
#                                 ', '.join(all_words.keys()))
#
#         # We need an arbitrary file to load as the input for the silence samples.
#         # It's multiplied by zero later, so the content doesn't matter.
#         silence_wav_path = self.data_buckets['training'][0]['file']
#
#         for set_index in ['validation', 'testing', 'training']:
#             set_size = len(self.data_buckets[set_index])
#             silence_size = int(math.ceil(set_size * silence_percentage / 100))
#             for _ in range(silence_size):
#                 self.data_buckets[set_index].append({
#                     'label': self._config.silence_label,
#                     'file': silence_wav_path
#                 })
#             # Pick some unknowns to add to each partition of the data set.
#             random.shuffle(unknown_buckets[set_index])
#             unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
#             self.data_buckets[set_index].extend(unknown_buckets[set_index][:unknown_size])
#
#         # Make sure the ordering is random.
#         for set_index in ['validation', 'testing', 'training']:
#             random.shuffle(self.data_buckets[set_index])
#         # Prepare the rest of the result data structure.
#         self.words_list = prepare_words_list(wanted_words)
#
#
#         for word in all_words:
#             if word in wanted_words_index:
#                 self.word_to_index[word] = wanted_words_index[word]
#             else:
#                 self.word_to_index[word] = self._config.unknown_word_index
#
#         self.word_to_index[self._config.silence_label] = self._config.silence_index
#
#         print_info(self.word_to_index)
#
#
#     def prepare_background_data(self):
#         """
#         Searches a folder for background noise audio, and loads it into memory.
#         It's expected that the background audio samples will be in a subdirectory
#         named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
#         the sample rate of the training data, but can be much longer in duration.
#         If the '_background_noise_' folder doesn't exist at all, this isn't an
#         error, it's just taken to mean that no background noise augmentation should
#         be used. If the folder does exist, but it's empty, that's treated as an
#         error.
#         :return: List of raw PCM-encoded audio samples of background noise.
#         :raises: Exception: If files aren't found in the folder.
#         """
#
#         background_dir = os.path.join(self.data_dir, self._config.background_noise_dir_name)
#         if not os.path.exists(background_dir):
#             return self.background_data
#
#         with tf.Session(graph=tf.Graph()) as sess:
#             wav_filename_placeholder = tf.placeholder(tf.string, [])
#             wav_loader = io_ops.read_file(wav_filename_placeholder)
#             wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
#             search_path = os.path.join(self.data_dir, self._config.background_noise_dir_name,
#                                        '*.wav')
#             for wav_path in gfile.Glob(search_path):
#                 wav_data = sess.run(
#                     wav_decoder,
#                     feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
#                 self.background_data.append(wav_data)
#             if not self.background_data:
#                 raise Exception('No background wav files were found in ' + search_path)
#
#
#     def get_train_files(self):
#         print_info("Number of trianing samples:  {}".format(len(self.data_buckets["training"])))
#         return self.data_buckets["training"]
#
#     def get_val_files(self):
#         print_info("Number of validation samples:  {}".format(len(self.data_buckets["validation"])))
#         return self.data_buckets["validation"]
#
#     def get_test_files(self):
#         print_info("Number of testing samples:  {}".format(len(self.data_buckets["testing"])))
#         return self.data_buckets["testing"]
#
#     def get_background_data(self):
#         return self.background_data