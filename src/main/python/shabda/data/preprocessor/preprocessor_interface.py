

class IPreprocessor():
    def __init__(self,
                 train_dir,
                 test_dir
                 ):
        pass

    def get_train_files(self):
        raise NotImplementedError

    def get_test_files(self):
        raise NotImplementedError

    def get_val_files(self):
        raise NotImplementedError

    def get_background_data(self):
        raise NotImplementedError
