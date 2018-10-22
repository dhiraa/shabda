import numpy as np
import librosa

def pad_data_array(data, max_audio_length):
    # Random offset / Padding
    if len(data) > max_audio_length:
        max_offset = len(data) - max_audio_length
        offset = np.random.randint(max_offset)
        data = data[offset:(max_audio_length + offset)]
    else:
        if max_audio_length > len(data):
            max_offset = max_audio_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, max_audio_length - len(data) - offset), "constant")
    return data


def get_frequency_spectrum(data, sampling_rate, n_mfcc):
    return librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc, n_fft=1024,
                                hop_length=345)  # with sr=44KHz and hop_length=345 to get n_mfcc x 256


def load_wav_audio_file(file_path):
    data, sample_rate = librosa.core.load(file_path, sr=None, res_type="kaiser_fast")
    # data, sample_rate = librosa.core.load("data/freesound-audio-tagging/input/audio_train/00044347.wav", res_type="kaiser_fast")
    return data, sample_rate


def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 1e-6)
    return data - 0.5