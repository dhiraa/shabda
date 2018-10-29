import unittest
import os
import numpy as np
from overrides import overrides
from shabda.data.dataset.internal.audio_dataset_base import AudioDatasetBase


class TestAudioDataset(AudioDatasetBase):
    def __init__(self, hparams):
        AudioDatasetBase.__init__(self, hparams=hparams)

    @overrides
    def init(self):
        self._setup_labels()

    @overrides
    def get_dataset_name(self):
        return "testdataset"

    @overrides
    def get_num_train_samples(self):
        return 0

    @staticmethod
    def default_hparams():
        return {"labels_index_map_store_path" : "/tmp/shabda/"}

    @overrides
    def get_default_labels(self):
        return ["_silence_", "_background_noise_"]

    @overrides
    def get_lables(self):
        return ["aone", "btwo", "cthree"]

class TestDatabaseBase(unittest.TestCase):
    def setUp(self):
        self.dataset = TestAudioDataset({"dummy_key": -1})
        self.dataset.init()

    def test_get_one_hot_encoded(self):

        vector = self.dataset.get_one_hot_encoded("aone")
        assert (np.array_equal(vector, [0,0,0,1,0,0]))

        vector = self.dataset.get_one_hot_encoded("cthree")
        assert (np.array_equal(vector, [0,0,0,0,0,1]))

        vector = self.dataset.get_one_hot_encoded("random_text")
        assert (np.array_equal(vector, [1, 0, 0, 0, 0, 0]))

        vector = self.dataset.get_one_hot_encoded("_silence_")
        vec = [0, 0, 0, 0, 0, 0]
        index = self.dataset.get_label_2_index("_silence_")
        vec[index] = 1
        assert (np.array_equal(vector, vec))

    def test_store_labels_index_map(self):
        stored_file_path = "/tmp/shabda/testdataset/labels_index_map.json"
        self.dataset.store_labels_index_map()

        file_exists = os.path.exists(stored_file_path)
        assert file_exists == True

        self.dataset.load_labels_index_map(stored_file_path)

        vector = self.dataset.get_one_hot_encoded("cthree")
        assert (np.array_equal(vector, [0, 0, 0, 0, 0, 1]))

if __name__ == '__main__':
    unittest.main()