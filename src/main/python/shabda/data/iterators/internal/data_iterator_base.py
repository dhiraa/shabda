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
Iterator that creates features for LSTM based models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from shabda.hyperparams.hyperparams import HParams

class DataIteratorBase():
    """

    """
    def __init__(self, hparams, dataset):
        self._hparams = HParams(hparams, default_hparams=self.get_default_params())
        self._dataset = dataset

    @staticmethod
    def get_default_params():
        return {"key": "value"}

    def _get_train_input_fn(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        raise NotImplementedError

    def _get_val_input_fn(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        raise NotImplementedError

    def _get_test_input_function(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        raise NotImplementedError

    def get_train_input_fn(self):
        """
        Returns an data set iterator function that can be used in estimator
        :return:
        """
        return self._get_train_input_fn()

    def get_val_input_fn(self):
        """
        Returns an data set iterator function that can be used in estimator
        :return:
        """
        return self._get_val_input_fn()

    def get_test_input_function(self):
        """
        Returns an data set iterator function that can be used in estimator
        :return:
        """
        return self._get_test_input_function()
