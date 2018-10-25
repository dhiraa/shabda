import sys
sys.path.append("../")


class DataIterator():

    def __init__(self):

        # Use one of the avaialble feature types in tc_utils.feature_types
        self._feature_type = None

        self.train_input_fn = None
        self.train_input_hook = None

        self.val_input_fn = None
        self.val_input_hook = None

        self.test_input_fn = None
        self.test_input_hook = None

    def prepare_train_set(self):
        '''
        Implement this function with reuqired TF function callbacks and hooks.
        :return:
        '''

        raise NotImplementedError

    def prepare_val_set(self):
        '''
        Implement this function with reuqired TF function callbacks and hooks.
        :return:
        '''

        raise NotImplementedError

    def prepare_test_set(self):
        '''
        Implement this function with reuqired TF function callbacks and hooks.
        :return:
        '''

        raise NotImplementedError

    def predict_on_csv_files(self, estimator):
        raise NotImplementedError

    def get_train_function(self):
        if self.train_input_fn is None:
            self.prepare_train_set()
        return self.train_input_fn

    def get_train_hook(self):
        if self.train_input_hook is None:
            self.prepare_train_set()
        return self.train_input_hook

    def get_val_function(self):
        if self.val_input_fn is None:
            self.prepare_val_set()
        return self.val_input_fn

    def get_val_hook(self):
        if self.val_input_hook is None:
            self.prepare_val_set()
        return self.val_input_hook

    def get_test_function(self):
        if self.test_input_fn is None:
            self.prepare_test_set()
        return self.test_input_fn

    def get_test_hook(self):
        if self.test_input_hook is None:
            self.prepare_test_set()
        return self.test_input_hook