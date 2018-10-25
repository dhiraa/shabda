import os
import re
import numpy as np
from scipy.io import wavfile
from glob import glob
from tqdm import tqdm
# it's a magic function :)
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
from sarvam.helpers.print_helper import *
from speech_recognition.sr_config.sr_config import *

class SimpleSpeechPreprocessor():
    def __init__(self,
                 data_dir,
                 possible_speech_commands,
                 batch_size):

        self.BATCH_SIZE = batch_size
        self.DATA_DIR = data_dir
        self.POSSIBLE_SPEECH_COMMANDS = prepare_words_list(possible_speech_commands)

        self.ID2NAME = {i: name for i, name in enumerate(self.POSSIBLE_SPEECH_COMMANDS)}
        # self.NAME2ID = {name: i for i, name in self.ID2NAME.items()}
        self.word_to_index = {name: i for i, name in self.ID2NAME.items()}


        self.TRAIN_SET, self.VALSET = self.load_data(self.DATA_DIR, self.POSSIBLE_SPEECH_COMMANDS)

        self.NUM_SAMPLES = len(self.TRAIN_SET)

    def load_data(self, data_dir, possible_labels):
        """ Return 2 lists of tuples:
        [(class_id, user_id, path), ...] for train
        [(class_id, user_id, path), ...] for validation
        """
        # Just a simple regexp for paths with three groups:
        # prefix, label, user_id
        pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
        all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

        with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
            validation_files = fin.readlines()

        valset = set()
        for entry in validation_files:
            r = re.match(pattern, entry)
            if r:
                valset.add(r.group(3))

        possible = set(possible_labels)

        train, val = [], []
        for entry in all_files:
            r = re.match(pattern, entry)
            if r:
                label, uid = r.group(2), r.group(3)
                if label == '_background_noise_':
                    label = SILENCE_LABEL
                if label not in possible:
                    label = UNKNOWN_WORD_LABEL

                # label_id = word_to_index[label]

                sample = {"label" : label, "file" : entry}
                if uid in valset:
                    val.append(sample)
                else:
                    train.append(sample)

        print_info('There are {} train and {} val samples'.format(len(train), len(val)))
        return train, val

    def get_train_files(self):
        print_info("Number of trianing samples:  {}".format(len(self.TRAIN_SET)))
        return self.TRAIN_SET

    def get_val_files(self):
        print_info("Number of validation samples:  {}".format(len(self.VALSET)))
        return self.VALSET

    def get_test_files(self):
        return self.data_buckets["testing"]
