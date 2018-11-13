"""
Convolutional Network Audio Classiifier default config.
"""

import copy
import tensorflow as tf

experiments = {
    "num_epochs": 15,
    "model_directory": "models/cnn_naive_classifier/",
    "dataset_name": "freesound_dataset",
    "data_iterator_name": "mfcc_iterator",
    "model_name": "cnn_naive_model",
    "learning_rate": 0.0001,

    "freesound_dataset": {
        "labels_index_map_store_path": "/tmp/shabda/",
        "n_classes": 41,
        "train_csv_path": "data/freesound-audio-tagging/input/train.csv",
        "val_csv_path": None,  # we dont have any validation csv file as such
        "test_csv_path": "./data/freesound-audio-tagging/input/sample_submission.csv",
        "train_audio_files_dir": "./data/freesound-audio-tagging/input/audio_train/",
        "val_audio_files_dir": "./data/freesound-audio-tagging/input/audio_train/",
        "test_audio_files_dir": "./data/freesound-audio-tagging/input/audio_test/"
    },

    "speech_recognition_dataest": {

        "data_dir": "./data/speech_commands_v0.02/",
        "data_url": "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        "max_num_wavs_per_class": 2 ** 27 - 1,  # ~134M
        "random_seed": 59185,
        "silence_label": '_silence_',
        "silence_index": 0,

        "unknown_word_label": '_unknown_',
        "unknown_word_index": 1,

        "background_noise_dir_name": '_background_noise_',
        # How much of the training data should be silence.
        "silence_percentage": 10.0,
        # How much of the training data should be unknown words.
        "unknown_percentage": 10.0,
        # What percentage of wavs to use as a test set.
        "testing_percentage": 10.0,
        # What percentage of wavs to use as a validation set.
        "validation_percentage": 10.0,

        # How loud the background noise should be, between 0 and 1.
        "background_volume": 0.1,
        # How many of the training samples have background noise mixed in.
        "background_frequency": 0.8,

        # Range to randomly shift the training audio by in time.
        "time_shift_ms": 100.0,
        # Expected sample rate of the wavs
        "sample_rate": 16000,
        # Expected duration in milliseconds of the wavs
        "clip_duration_ms": 1000,
        # How long each spectrogram timeslice is
        "window_size_ms": 30.0,
        # How long each spectrogram timeslice is
        "window_stride_ms": 10.0,
        # How many bins to use for the MFCC fingerprint
        "dct_coefficient_count": 40,
        # How many training loops to run
        "how_many_training_steps": 15000,
        # How often to evaluate the training results.
        "eval_step_interval": 400,
        # Words to use (others will be added to an unknown label)
        # wanted_words : yes,no,up,down,left,right,on,off,stop,go
        "possible_commands": "yes,no,up,down,left,right,on,off,stop,go",

        # "TIME_SHIFT_SAMPLES" : int((TIME_SHIFT_MS * SAMPLE_RATE) / 1000)
    },

    "data_iterator": {
        "use_mfcc": False,
        "n_mfcc": 64,
        "batch_size": 32,
        "sampling_rate": 44100,
        "audio_duration": 2,
    },

    "model": {
        "out_dim": 41 + 1,  # one unknown
        "name": "cnn_naive",
        "learning_rate": 0.001
    },

}
