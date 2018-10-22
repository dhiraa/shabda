import sys
sys.path.append("../")

from importlib import import_module
import shabda.dataset
class DatasetFactory():

    dataset_path = {
        "freesound_dataset": "shabda.dataset.freesound_dataset",
    }

    datasets = {
        "freesound_dataset": "FreeSoundDataset",
    }

    def __init__(self):
        pass

    @staticmethod
    def _get_dataset(name):
        try:
            dataset = getattr(import_module(DatasetFactory.dataset_path[name]), DatasetFactory.datasets[name])
        except KeyError:
            raise NotImplemented("Given dataset file name not found: {}".format(name))
        # Return the model class
        return dataset

    @staticmethod
    def get(dataset_name):
        dataset = DatasetFactory._get_dataset(dataset_name)
        return dataset


