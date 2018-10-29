from shabda.data.iterators.internal.data_iterator_base import DataIteratorBase
from shabda.data.dataset.internal.audio_dataset_base import AudioDatasetBase

class FreeSoundDataIteratorBase(DataIteratorBase):
    def __init__(self, hparams, dataset: AudioDatasetBase):
        DataIteratorBase.__init__(self, hparams, dataset)

        if not isinstance(dataset,  AudioDatasetBase):
            raise AssertionError("dataser should be an inherited class of AudioDatasetBase")

        self.name = "FreeSoundDataIteratorBase"

        self._batch_size = self._hparams.batch_size
        self._num_epochs = self._hparams.num_epochs
        self._sampling_rate = self._hparams.sampling_rate
        self._max_audio_length = self._hparams.sampling_rate * self._hparams.audio_duration  # 32000
        self._n_mfcc = self._hparams.n_mfcc

        self._train_files_path = dataset.get_train_files()
        self._train_labels = dataset.get_train_labels()

        self._val_files_path = dataset.get_val_files()
        self._val_labels = dataset.get_val_labels()

        self.test_files_path = []

    @staticmethod
    def get_default_params():
        config = {"use_mfcc": False,
        "n_mfcc": 64,
        "batch_size": 64,
        "num_epochs" : 1,
        "sampling_rate": 44100,
        "audio_duration": 2}

        return config

    def _get_train_input_fn(self):
        raise NotImplementedError

    def _get_val_input_fn(self):
        raise NotImplementedError

    def _get_test_input_function(self):
        raise NotImplementedError