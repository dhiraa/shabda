from shabda.hyperparams.hyperparams import HParams

class DataIteratorBase():
    def __init__(self, hparams, dataset):
        self._hparams = HParams(hparams, default_hparams=None)
        self._dataset = dataset

    def _get_train_input_fn(self):
        raise NotImplementedError

    def _get_val_input_fn(self):
        raise NotImplementedError

    def _get_test_input_function(self):
        raise NotImplementedError

    def get_train_input_fn(self):
        return self._get_train_input_fn()

    def get_val_input_fn(self):
        return self._get_val_input_fn()

    def get_test_input_function(self):
        return self._get_test_input_function()
