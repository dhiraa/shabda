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
Experiments class that allows easy plug n play of modules
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
from tqdm import tqdm
import tensorflow as tf
import logging

from shabda.data.dataset.internal.audio_dataset_base import AudioDatasetBase
from shabda.data.dataset.internal.dataset_factory import DatasetFactory
from shabda.data.iterators.internal.data_iterator_factory import DataIteratorFactory
from shabda.data.iterators.internal.data_iterator_base import DataIteratorBase
from shabda.models.internal.model_factory import ModelsFactory
# TODO - "import *" also import some python meta variables, hence not recommended
from shabda.helpers.print_helper import *
from shabda.hyperparams.hyperparams import HParams
from shabda.run.executor import Executor

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

CGREEN2 = '\33[92m'
CEND = '\33[0m'


class Experiments(object):
    """
    Experiments uses dataset, data iterator & model factory classes and import them
    dynamically based on the string.
    This allows the user to choose the modules dynamically and run the experiments without ever writing the
    code when we need mix and experiment dataset and modules.
    """

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
        dataset = DatasetFactory.get(dataset_file_name=dataset_name)
        return dataset

    def get_iterator_reference(self, iterator_name):
        """
        Uses the iterator name to get the reference from the iterator factory class

        :param iterator_name:
        :return:
        """

        print_debug("Geting iterator  :" + iterator_name)
        iterator = DataIteratorFactory.get(iterator_name=iterator_name)
        return iterator

    def get_model_reference(self, model_name):
        """
        Uses the model name to get the reference from the model factory class

        :param model_name:
        :return:
        """

        print_debug("Geting model :" + model_name)
        model = ModelsFactory.get(model_name=model_name)
        return model

    def setup(self):

        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        # run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
        run_config.allow_soft_placement = True
        run_config.log_device_placement = False
        self._run_config = tf.contrib.learn.RunConfig(session_config=run_config,
                                                      save_checkpoints_steps=50,
                                                      keep_checkpoint_max=5,
                                                      save_summary_steps=25,
                                                      model_dir=self._hparams["model_directory"],
                                                      log_step_count_steps=10)

        # Using factory classes get the handle for the actual classes from string
        self.dataset = self.get_dataset_reference(self._hparams.dataset_name)
        self._data_iterator = self.get_iterator_reference(self._hparams.data_iterator_name)
        self.model = self.get_model_reference(self._hparams.model_name)

        # Initialize the handles and call any user specific init() methods
        self.dataset: AudioDatasetBase = self.dataset(hparams=self._hparams[self._hparams['dataset_name']])
        self.dataset.init()

        self._data_iterator: DataIteratorBase = self._data_iterator(hparams=self._hparams.data_iterator,
                                                                    dataset=self.dataset)

        model_params = self._hparams.model

        self.model = self.model(hparams=model_params)

    def test_dataset(self):
        iterator = self._data_iterator.get_train_input_fn().make_initializable_iterator()
        next_element = iterator.get_next()
        print_error(next_element)
        init_op = iterator.initializer
        with tf.Session() as sess:
            # Initialize the iterator
            sess.run(init_op)
            print(sess.run(next_element))
            print(sess.run(next_element))
            # Move the iterator back to the beginning
            sess.run(init_op)
            print(sess.run(next_element))

    def run(self):
        self.setup()
        num_samples = self.dataset.get_num_train_samples()
        batch_size = self._hparams.data_iterator.batch_size
        num_epochs = self._hparams.num_epochs
        mode = self.mode

        exec = Executor(model=self.model, data_iterator=self._data_iterator, config=self._run_config)

        if (mode == "train") or (mode == "retrain"):
            for current_epoch in tqdm(range(num_epochs), desc="Epoch"):
                current_max_steps = (num_samples // batch_size) * (current_epoch + 1)
                exec.train(max_steps=current_max_steps)  # , eval_steps=None)
                exec.evaluate(steps=200)
