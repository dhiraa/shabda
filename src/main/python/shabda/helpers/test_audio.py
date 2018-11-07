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
Utilities for audio related operations.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import unittest
import numpy as np
from shabda.helpers.audio import pad_data_array, audio_norm

class TestAudioUtility(unittest.TestCase):
    """

    """
    def test_pad_data_array(self):
        array = np.ones(12)
        array_filled = pad_data_array(array, 20)
        assert array_filled.shape[0] == 20


    def test_audio_norm(self):
        arr = np.array([1, 2, 1, 2, 1])
        arr_normalised = audio_norm(arr)
        expected = np.array([-0.5, 0.499999, -0.5, 0.499999, -0.5])
        np.testing.assert_allclose(arr_normalised, expected)


if __name__ == '__main__':
    unittest.main()