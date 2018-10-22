from shabda.iterators.internal.data_iterator_base import DataIteratorBase


class FreeSoundDataIteratorBase(DataIteratorBase):
    def __init__(self, hparams, dataset):
        DataIteratorBase.__init__(self, hparams, dataset)

        self.name = "FreeSoundDataIteratorBase"

        self._batch_size = self._hparams.batch_size
        self._sampling_rate = self._hparams.sampling_rate
        self._max_audio_length = self._hparams.sampling_rate * self._hparams.audio_duration  # 32000
        self._n_mfcc = self._hparams.n_mfcc

        self.train_wav_files_path = []
        self.val_wav_files_path = []
        self.test_wav_files_path = []


        self.train_labels = dataset.get_dev_labels_df().values
        self.val_labels = dataset.get_val_labels_df().values

        for file_name in dataset.get_dev_wav_files():
            self.train_wav_files_path.append(dataset.dev_audio_files_dir + "/" + file_name)

        for file_name in dataset.get_val_wav_files():
            self.val_wav_files_path.append(dataset.dev_audio_files_dir + "/" + file_name)

        for file_name in dataset.get_val_wav_files():
            self.test_wav_files_path.append(dataset.dev_audio_files_dir + "/" + file_name)


    def _get_train_input_fn(self):
        raise NotImplementedError

    def _get_val_input_fn(self):
        raise NotImplementedError

    def _get_test_input_function(self):
        raise NotImplementedError