import numpy as np

class HyperParams(object):
    def __init__(self,
                 sampling_rate=44100,
                 audio_duration=2,
                 n_classes=41,
                 use_mfcc=False,
                 n_folds=10,
                 learning_rate=0.0001,
                 batch_size=4,
                 num_epochs=5,
                 n_mfcc=64):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.max_audio_length = self.sampling_rate * self.audio_duration # 32000

        # if self.use_mfcc:
        #     self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        # else:
        #     self.dim = (self.audio_length, 1)