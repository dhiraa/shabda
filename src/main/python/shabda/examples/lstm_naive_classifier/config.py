"""
Convolutional Network Audio Classiifier default config.
"""

import copy
import tensorflow as tf

experiments = {
    "num_epochs": 15,
    "model_directory": "models/naive_lstm/",
    "dataset_name": "freesound_dataset",
    "data_iterator_name": "lstm_features",
    "model_name": "naive_lstm",
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

    "data_iterator": {
        "use_mfcc": False,
        "n_mfcc": 64,
        "batch_size": 32,
        "sampling_rate": 44100,
        "audio_duration": 2,
    },

    "cnn_naive": {
        "out_dim": 41 + 1,  # one unknown
        "name": "cnn_naive",
        "learning_rate": 0.001
    },

    "model": {
        "out_dim": 41 + 1,  # one unknown
        "name": "lstm_naive",
        "learning_rate": 0.001
    }
}
