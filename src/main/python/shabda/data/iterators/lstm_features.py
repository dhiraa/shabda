import numpy as np
import librosa
import tensorflow as tf


def append_zeros(required_dim, data, is_row=False):
    rows = data.shape[0]
    cols = data.shape[1]

    new_rows = required_dim - rows
    new_cols = required_dim - cols
    if is_row:
        if rows > required_dim:
            return data
        return np.vstack([data, np.zeros(shape=(new_rows, cols))])
    else:
        if cols > required_dim:
            return data
        return np.hstack([data, np.zeros(shape=(rows, new_cols))])


# References: https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification
def extract_audio_features(audio_file, hop_length):
    # timeseries_length = min(self.timeseries_length_list)
    timeseries_length = 128
    data = np.zeros((1, timeseries_length, 33), dtype=np.float64)
    target = []

    y, sr = librosa.load(audio_file, duration=3.0)
    y, index = librosa.effects.trim(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    mfcc = append_zeros(timeseries_length, mfcc)
    print(mfcc.T.shape)

    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spectral_center = append_zeros(timeseries_length, spectral_center)
    print(spectral_center.T.shape)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    chroma = append_zeros(timeseries_length, chroma)
    print(chroma.T.shape)

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    spectral_contrast = append_zeros(timeseries_length, spectral_contrast)
    print(spectral_contrast.T.shape)

    data[:, :, 0:13] = mfcc.T[0:timeseries_length, :]
    data[:, :, 13:14] = spectral_center.T[0:timeseries_length, :]
    data[:, :, 14:26] = chroma.T[0:timeseries_length, :]
    data[:, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

    #     print("Extracted features audio track %s." % audio_file)

    return data  # , np.expand_dims(np.asarray(target), axis=1)

class AudioIterator(object):
    def __init__(self,
                 train_files,
                 train_labels,
                 val_files,
                 val_labels,
                 test_files):
        self._train_files = train_files
        self._train_labels = train_labels

        self._val_files = val_files
        self._val_labels = val_labels

        self._test_files = test_files

    def get_train_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self._train_files, self._train_labels))
        dataset = dataset.shuffle(len(self._train_labels))
        dataset = dataset.map(self.train_preprocess, num_parallel_calls=4)
        dataset = dataset.batch(32)
        dataset = dataset.prefetch(1)
        return dataset

    def train_preprocess(self, filename, label):
        return extract_audio_features(audio_file=filename), label

    def get_train_iterator(self):
        dataset = self.get_train_dataset()
        iterator = dataset.make_one_shot_iterator()
        return iterator

