import numpy as np
import json
import os

from shabda.hyperparams.hyperparams import HParams
from shabda.helpers import utils

class DatasetBase(object):
    def __init__(self, hparams):
        dataset_hparams = utils.dict_fetch(
            hparams, self.default_hparams())

        self._hparams = HParams(dataset_hparams, self.default_hparams())

        self._labels_2_index = None
        self._index_2_labels = None

    def _setup_labels(self):
        self._labels = self.get_lables()
        self._labels = self.get_default_labels() + list(self._labels)
        self._labels = sorted(self._labels)

        self._labels_2_index = {label.lower():i for i, label in enumerate(self._labels, 1)}
        self._index_2_labels = {i: label.lower() for label, i in self._labels_2_index.items()}

        self._unknown_label = "_unknown_"
        self._labels_2_index[ self._unknown_label] = 0
        self._index_2_labels[0] =  self._unknown_label

        self._labels_length = len(self._labels_2_index)

    def init(self):
        raise NotImplementedError

    @staticmethod
    def default_hparams():
        params = {
            "labels_index_map_store_path" : "/tmp/shabda/"
        }
        return params

    def get_labels_dim(self):
        return self._labels_length

    def get_dataset_name(self):
        return "dummy_dataset"

    def get_num_train_samples(self):
        """

        :return: int Number of training sample for current dataset
        """
        raise NotImplementedError

    def get_default_labels(self):
        """

        :return: list of default lables. Eg: ["_silence_", "_background_noise_"]
        """
        raise NotImplementedError

    def get_lables(self):
        """

        :return: List of lables for the dataset under consideration
        """
        raise NotImplementedError

    def get_label_2_index(self, label):
        return self._labels_2_index.get(label, 0) #return unknown index when not found

    def get_index_2_label(self, index):
        return self._index_2_labels.get(index,  self._unknown_label)

    def get_one_hot_encoded(self, label):
        label = label.lower()
        vector = np.zeros(self._labels_length)
        index = self.get_label_2_index(label=label)
        vector[index] = 1
        return vector

    def store_labels_index_map(self):
        dir = self._hparams["labels_index_map_store_path"] + "/" + self.get_dataset_name() + "/"
        file_name = "labels_index_map.json"

        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(dir + file_name, 'w') as fp:
            json.dump(self._labels_2_index, fp)

    def load_labels_index_map(self, file_path):
        with open(file_path) as handle:
            self._labels_2_index = json.loads(handle.read())
            self._index_2_labels = {i: label.lower() for label, i in self._labels_2_index.items()}
            self._labels_length = len(self._labels_2_index)
