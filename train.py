"""Train the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_fn import Reader
from utils.utils import Params
from utils.utils import set_logger
from model.model_fn import model_fn
from model.training import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='input',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")
parser.add_argument('--train_data_dir', default='input/train',
                    help="train dataset dir")
parser.add_argument('--dev_data_dir', default='input/dev',
                    help="dev dataset dir")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params_video_level.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)


    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(
        os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = args.train_data_dir
    dev_data_dir = args.dev_data_dir

    # Get the filenames from the train and dev sets
    train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)]
    dev_filenames = [os.path.join(dev_data_dir, f) for f in os.listdir(dev_data_dir)]

    # Create the two iterators over the two datasets
    train_reader = Reader(True, train_filenames, params)
    dev_reader = Reader(False, dev_filenames, params)
    train_inputs = train_reader.input_fn()
    dev_inputs = dev_reader.input_fn()

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', dev_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    trainer = Trainer(train_model_spec, eval_model_spec,
                      args.model_dir, params, args.restore_from)
    trainer.train_and_evaluate()

