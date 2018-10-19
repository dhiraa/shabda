import argparse
import tensorflow as tf
import sys
import os
sys.path.append("src/main/python/")

from shabda.hparams import HyperParams
from shabda.dataset import FreeSoundDataset
from shabda.iterator import DataIterator
#TODO Changed here from cnn_naive_model
from shabda.cnn_beginners_model import CustomDNN
import logging
from shabda.helpers.print_helper import *

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

CGREEN2  = '\33[92m'
CEND      = '\33[0m'

def run(opt):

    num_epochs = opt.num_epochs
    batch_size = opt.batch_size

    if not os.path.exists(opt.model_dir):
        os.makedirs(opt.model_dir)

    hparams = HyperParams(batch_size=batch_size, num_epochs=num_epochs)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    # run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
    run_config.allow_soft_placement = True
    run_config.log_device_placement = False
    run_config = tf.contrib.learn.RunConfig(session_config=run_config,
                                            save_checkpoints_steps=50,
                                            keep_checkpoint_max=5,
                                            save_summary_steps=25,
                                            model_dir=opt.model_dir)

    dataset = FreeSoundDataset()
    dataset.load_data_info()
    data_iterator = DataIterator(hparams)
    model = CustomDNN(model_dir=opt.model_dir, hparams=hparams, run_config=run_config)

    train_wav_files_path = []
    val_wav_files_path = []

    train_labels = dataset.get_dev_labels_df().values
    val_labels = dataset.get_val_labels_df().values

    for file_name in dataset.get_dev_wav_files():
        train_wav_files_path.append(dataset.dev_audio_files_dir + "/" + file_name)


    for file_name in dataset.get_val_wav_files():
        val_wav_files_path.append(dataset.dev_audio_files_dir + "/" + file_name)

    num_samples = dataset.get_num_samples()

    print(len(val_wav_files_path))
    print_info(num_samples)
    print_info(batch_size)
    max_steps = (num_samples // batch_size) * (0 + 1)
    print_info(max_steps)

    if (opt.mode == "train" or opt.mode == "retrain"):
        for current_epoch in range(num_epochs):

            max_steps = (num_samples // batch_size) * (current_epoch + 1)

            model.train(input_fn=data_iterator.get_train_input_fn(train_wav_files_path, train_labels),
                        max_steps=max_steps)

            tf.logging.info(CGREEN2 + str("Evalution on epoch: " + str(current_epoch + 1)) + CEND)

            eval_results = model.evaluate(input_fn=data_iterator.get_val_input_fn(val_wav_files_path=val_wav_files_path,
                                                                                  labels=val_labels))

            tf.logging.info(CGREEN2 + str(str(eval_results)) + CEND)
    elif opt.mode == "predict":
        dataset.predict_on_test_files(data_iterator, model)



if __name__ == "__main__":
    optparse = argparse.ArgumentParser("Run experiments on available models and datasets")


    optparse.add_argument('-mode', '--mode',
                          choices=['train', "retrain", "predict"],
                          required=True,
                          help="'train', 'retrain','predict'"
                          )

    optparse.add_argument('-md', '--model-dir', action='store',
                          dest='model_dir',
                          required=True,
                          help='Model directory needed for training')

    optparse.add_argument('-dsn', '--dataset-name', action='store',
                          dest='dataset_name', required=False,
                          help='Name of the Dataset to be used')

    optparse.add_argument('-din', '--data-iterator-name', action='store',
                          dest='data_iterator_name', required=False,
                          help='Name of the DataIterator to be used')

    optparse.add_argument('-bs', '--batch-size',  type=int, action='store',
                          dest='batch_size', required=False,
                          default=8,
                          help='Batch size for training, be consistent when retraining')

    optparse.add_argument('-ne', '--num-epochs', type=int, action='store',
                          dest='num_epochs',
                          default=5,
                          required=False,
                          help='Number of epochs')

    optparse.add_argument('-mn', '--model-name', action='store',
                          dest='model_name', required=False,
                          help='Name of the Model to be used')

    opt = optparse.parse_args()
    if (opt.mode == 'retrain' or opt.mode == 'predict') and not opt.model_dir:
        optparse.error('--model-dir argument is required in "retrain" & "predict" mode.')
    else:
        run(opt)