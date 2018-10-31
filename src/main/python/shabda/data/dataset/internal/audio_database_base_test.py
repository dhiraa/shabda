# Copyright 2018 The Shabda Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test cases for AudioDatasetBase
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import unittest

from overrides import overrides
from shabda.data.dataset.internal.audio_dataset_base import AudioDatasetBase


class TestAudioDataset(AudioDatasetBase):
    """
    A test dataset emulating the labels
    """
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
    def get_predefined_labels(self):
        return ["_silence_", "_background_noise_"]

    @overrides
    def get_labels(self):
        return ["aone", "btwo", "cthree"]

class TestDatabaseBase(unittest.TestCase):
    """
    Test cases for AudioDatasetBase
    """
    def setUp(self):
        self.dataset = TestAudioDataset(hparams=None)
        self.dataset.init()

    def test_get_one_hot_encoded(self):
        """
        Test case for `get_one_hot_encoded`
        :return:
        """

        vector = self.dataset.get_one_hot_encoded("aone")
        assert (np.array_equal(vector, [0, 0, 0, 1, 0, 0]))

        vector = self.dataset.get_one_hot_encoded("cthree")
        assert (np.array_equal(vector, [0, 0, 0, 0, 0, 1]))

        vector = self.dataset.get_one_hot_encoded("random_text")
        assert (np.array_equal(vector, [1, 0, 0, 0, 0, 0]))

        vector = self.dataset.get_one_hot_encoded("_silence_")
        vec = [0, 0, 0, 0, 0, 0]
        index = self.dataset.get_label_2_index("_silence_")
        vec[index] = 1
        assert (np.array_equal(vector, vec))

    def test_store_labels_index_map(self):
        """
        Test case for `store_labels_index_map`
        :return:
        """
        stored_file_path = "/tmp/shabda/testdataset/labels_index_map.json"
        self.dataset.store_labels_index_map()
        exists = os.path.exists(stored_file_path)
        assert exists

        self.dataset.load_labels_index_map(stored_file_path)

        vector = self.dataset.get_one_hot_encoded("cthree")
        assert (np.array_equal(vector, [0, 0, 0, 0, 0, 1]))

if __name__ == '__main__':
    unittest.main()