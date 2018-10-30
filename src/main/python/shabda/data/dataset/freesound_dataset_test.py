# import tensorflow as tf
# from shabda.data.dataset.freesound_dataset import FreeSoundAudioDataset
#
#
# class FreeSoundDatasetTest(tf.test.TestCase):
#
#
#     def test_init(self):
#         dataset = FreeSoundAudioDataset()
#         dataset.init()
#         assert (dataset.train_df.shape[0] != 0)
#         assert (dataset.val_df.shape[0] != 0)
#         assert (dataset.test_df.shape[0] != 0)
#
#     def test_get_num_samples(self):
#         dataset = FreeSoundAudioDataset()
#         dataset.init()
#         assert (dataset.train_df.shape[0] != 0)
#
#     def test_load_train_audio_file(self):
#         dataset = FreeSoundAudioDataset()
#         dataset.init()
#         file_name = dataset.train_df["fname"][0]
#         data, sample_rate = FreeSoundAudioDataset.load_wav_audio_file(file_path=dataset.dev_audio_files_dir + "/" + file_name)
#         assert (data.shape[0] != 0)
#
#
#     def test_pad_data_array(self):
#         dataset = FreeSoundAudioDataset()
#         dataset.init()
#         file_name = dataset.train_df["fname"][0]
#         data, sample_rate = FreeSoundAudioDataset.load_wav_audio_file(file_path=dataset.dev_audio_files_dir + "/" + file_name)
#         data = FreeSoundAudioDataset.pad_data_array(data=data, max_audio_length=20000)
#
#         assert (data.shape[0] == 20000)
#
#         data = FreeSoundAudioDataset.pad_data_array(data=data, max_audio_length=200)
#
#         assert (data.shape[0] == 200)
#
#
#     def test_get_frequency_spectrum(self):
#         dataset = FreeSoundAudioDataset()
#         dataset.init()
#         file_name = dataset.train_df["fname"][0]
#         data, sample_rate = FreeSoundAudioDataset.load_wav_audio_file(file_path=dataset.dev_audio_files_dir + "/" + file_name)
#         data = FreeSoundAudioDataset.pad_data_array(data=data, max_audio_length=44100 * 2) # 2 seconds
#
#         data = FreeSoundAudioDataset.get_frequency_spectrum(data=data, n_mfcc=128, sampling_rate=44100)
#         assert (data.shape[0] == 128)
#         assert (data.shape[1] == 256)
#
#
#     def test_get_train_labels(self):
#         dataset = FreeSoundAudioDataset()
#         dataset.init()
#
#         one_hot_encoded_labesl_df = dataset.get_dev_labels()
#
#         print(one_hot_encoded_labesl_df.head(5))
#         # we have 41 unique categories
#         assert (one_hot_encoded_labesl_df.values.shape[1] == 41)
#
#         #check the order of columns are same always
#         assert (one_hot_encoded_labesl_df.columns[0] == "Acoustic_guitar")
#         assert (one_hot_encoded_labesl_df.columns[1] == "Applause")
#
#
#         assert (one_hot_encoded_labesl_df.columns[39] == "Violin_or_fiddle")
#         assert (one_hot_encoded_labesl_df.columns[40] == "Writing")
#
#
#
#
