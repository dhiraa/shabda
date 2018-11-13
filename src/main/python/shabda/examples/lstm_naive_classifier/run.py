"""Example to run LSTM naive Audio Classifier

To run:
$ python src/main/python/shabda/examples/lstm_naive_classifier/run.py --mode=train

"""
import importlib
import sys
import os
import tensorflow as tf

sys.path.append("src/main/python/")
sys.path.append("../..")

from shabda.run.experiments import Experiments

tf.flags.DEFINE_string("config", "config", "The config to use.")
tf.flags.DEFINE_string("mode", "train", "train/retrain/predict")
FLAGS = tf.flags.FLAGS
config = importlib.import_module(FLAGS.config)

if __name__ == "__main__":
    experiment = Experiments(hparams=config.experiments, mode=FLAGS.mode)
    experiment.run()
