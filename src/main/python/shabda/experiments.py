import sys
import os
import argparse
from tqdm import tqdm
import tensorflow as tf
import logging


from shabda.data.dataset.internal.dataset_factory import DatasetFactory
from shabda.data.iterators.internal.data_iterator_factory import DataIteratorFactory
from shabda.data.iterators.internal.data_iterator_base import DataIteratorBase
from shabda.models.internal.model_factory import ModelsFactory
from shabda.helpers.print_helper import *
from shabda.hyperparams.hyperparams import HParams

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

CGREEN2  = '\33[92m'
CEND      = '\33[0m'

class Experiments():
    def __init__(self, hparams, mode='train'):
        self._hparams = HParams(hparams, self.default_hparams())

        self.mode = mode

        self.dataset = None
        self.data_iterator = None
        self.model = None

    @staticmethod
    def default_hparams():
        return None

    def get_dataset_reference(self, dataset_name):
        """
        Uses the dataset name to get the reference from the dataset factory class
        :param dataset_name:
        :return:
        """

        print_debug("Geting dataset :" + dataset_name)
        dataset =  DatasetFactory.get(dataset_name=dataset_name)
        return  dataset

    def get_iterator_reference(self, iterator_name):
        """
        Uses the iterator name to get the reference from the iterator factory class
        :param iterator_name:
        :return:
        """

        print_debug("Geting iterator  :" + iterator_name)
        iterator =  DataIteratorFactory.get(iterator_name=iterator_name)
        return  iterator

    def get_model_reference(self, model_name):
        """
        Uses the model name to get the reference from the model factory class
        :param model_name:
        :return:
        """

        print_debug("Geting model :" + model_name)
        model =  ModelsFactory.get(model_name=model_name)
        return  model

    def setup(self):
        self.dataset = self.get_dataset_reference(self._hparams.dataset_name)
        self.data_iterator  = self.get_iterator_reference(self._hparams.data_iterator_name)
        self.model = self.get_model_reference(self._hparams.model_name)

        self.dataset = self.dataset()
        self.data_iterator: DataIteratorBase  = self.data_iterator(hparams=self._hparams.data_iterator, dataset = self.dataset)

        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        # run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
        run_config.allow_soft_placement = True
        run_config.log_device_placement = False
        run_config = tf.contrib.learn.RunConfig(session_config=run_config,
                                                save_checkpoints_steps=50,
                                                keep_checkpoint_max=5,
                                                save_summary_steps=25,
                                                model_dir=self._hparams["model_directory"])

        self.model = self.model(model_dir=self._hparams.model_directory,
                                run_config=run_config,
                                hparams=None)

    def run(self):
        self.setup()
        num_samples = self.dataset.get_num_samples()
        batch_size = self._hparams.data_iterator.batch_size
        num_epochs = self._hparams.num_epochs
        mode = self.mode

        if (mode == "train" or mode == "retrain"):
            for current_epoch in tqdm(range(num_epochs)):
                current_max_steps = (num_samples // batch_size) * (current_epoch + 1)

                self.model.train(input_fn=self.data_iterator.get_train_input_fn(),
                            max_steps=current_max_steps)

                tf.logging.info(CGREEN2 + str("Evaluation on epoch: " + str(current_epoch + 1)) + CEND)

                eval_results = self.model.evaluate(input_fn=self.data_iterator.get_val_input_fn())

                tf.logging.info(CGREEN2 + str(str(eval_results)) + CEND)
        elif mode == "predict":
            self.dataset.predict_on_test_files()

