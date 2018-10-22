"""
Convolutional Network Audio Classiifier default config.
"""

import copy
import tensorflow as tf

experiments = {
    "num_epochs" : 15,
    "model_directory" : "models/cnn_naive_classifier/",
    "dataset_name" : "freesound_dataset",
    "data_iterator_name" : "mfcc_iterator",
    "model_name" : "cnn_naive_model",
    "learning_rate" : 0.0001,

    "freesound_dataset" : {
        "n_classes": 41,
        "dev_csv_path" : "data/freesound-audio-tagging/input/train.csv",
        "val_csv_path" : "None",  # we dont have any validation csv file as such
        "test_csv_path" : "./data/freesound-audio-tagging/input/sample_submission.csv",
        "dev_audio_files_dir" : "./data/freesound-audio-tagging/input/audio_train/",
        "val_audio_files_dir" : "./data/freesound-audio-tagging/input/audio_train/",
        "test_audio_files_dir" : "./data/freesound-audio-tagging/input/audio_test/"
    },

    "data_iterator" : {
        "use_mfcc": False,
        "n_mfcc": 64,
        "batch_size": 64,
        "sampling_rate": 44100,
        "audio_duration": 2,
    },

    "model" : {

    }
}