from shabda.hyperparams.hyperparams import HParams
from shabda.helpers import utils

class DatasetBase():
    def __init__(self, hparams):
        dataset_hparams = utils.dict_fetch(
            hparams, self.default_hparams())

        self._hparams = HParams(dataset_hparams, self.default_hparams())

    def get_num_samples(self):
        raise NotImplementedError

    @staticmethod
    def default_hparams():
        raise NotImplementedError