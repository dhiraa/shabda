"""Example to run LSTM naive Audio Classifier

To run:
$ python src/main/python/shabda/examples/lstm_naive_classifier/run.py --mode=train

"""
import importlib
import sys
import os
import tensorflow as tf
sys.path.append("src/main/python/")

from shabda.run.experiments import Experiments

flags = tf.flags
flags.DEFINE_string("config", "config", "The config to use.")
flags.DEFINE_string("mode", "train", "train/retrain/predict")
FLAGS = flags.FLAGS
config = importlib.import_module(FLAGS.config)

if __name__ == "__main__":
    experiment = Experiments(hparams=config.experiments, mode=FLAGS.mode)
    experiment.run()