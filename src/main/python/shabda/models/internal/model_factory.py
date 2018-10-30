import sys
sys.path.append("../")

from importlib import import_module

class ModelsFactory():

    model_path = {
        "cnn_beginners_model" : "shabda.models.cnn_beginners_model",
        "cnn_naive_model": "shabda.models.clasifiers.cnn_naive_model",
        "naive_lstm" : "shabda.models.clasifiers.naive_lstm"
    }

    models = {
        "cnn_beginners_model" :  "CustomDNN",
        "cnn_naive_model" : "CustomDNN",
        "naive_lstm" : "NaiveLSTM"
    }


    def __init__(self):
        pass

    @staticmethod
    def _get_model(name):

        try:
            model = getattr(import_module(ModelsFactory.model_path[name]), ModelsFactory.models[name])
        except KeyError:
            raise NotImplemented("Given config file name not found: {}".format(name))
        # Return the model class
        return model


    @staticmethod
    def get(model_name):
        model = ModelsFactory._get_model(model_name)
        return model


