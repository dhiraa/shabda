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
Factory class to import the datasets dynamically
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
from importlib import import_module

sys.path.append("../")


class DatasetFactory():
    """
    Factory class to import the datasets dynamically.
    This is used in conjuction with experiments, where dataset can be plugged in
    with the change of the filename
    """

    dataset_path = {
        # file_name : package
        "freesound_dataset": "shabda.data.dataset.freesound_dataset",
        "speech_commands_v0_02": "shabda.data.dataset.speech_commands_v0_02"
    }

    datasets = {
        # file_name : class_name
        "freesound_dataset": "FreeSoundAudioDataset",
        "speech_commands_v0_02": "SpeechCommandsV002"
    }

    @staticmethod
    def _get_dataset(name):
        """
        Finds the package and gets the class handle for the dataset file name
        :param name:
        :return:
        """
        try:
            dataset = getattr(import_module(DatasetFactory.dataset_path[name]), DatasetFactory.datasets[name])
        except KeyError:
            raise NotImplemented("Given dataset file name not found: {}".format(name))
        return dataset

    @staticmethod
    def get(dataset_file_name):
        """
        Returns the class handle for the given file name
        :param dataset_file_name:
        :return: class handle
        """
        dataset = DatasetFactory._get_dataset(dataset_file_name)
        return dataset
