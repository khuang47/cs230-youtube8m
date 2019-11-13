"""Tensorflow utility functions for training"""

import logging
import os
import tensorflow as tf
import metrics.eval_util as eval_util
from tensorflow.core.framework import summary_pb2


def train_sess(sess, model_spec, writer, params):
    """Train the model on `num_steps` batches

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        writer: (tf.summary.FileWriter) writer for summaries
        params: (Params) hyper-parameters
    """
    # Get relevant graph operations or nodes needed for training
    probabilities = model_spec['probabilities']
    labels = model_spec['labels']
    loss = model_spec['loss']
    train_op = model_spec['train_op']

    global_step = tf.train.get_global_step()
    sess.run(model_spec['iterator_init_op'])
    assert probabilities.get_shape().as_list() == [params.batch_size, params.vocab_size]
    assert labels.get_shape().as_list() == [params.batch_size, params.vocab_size]

    i = 0
    while True:
        try:
            # Evaluate summaries for tensorboard only once in a while
            if i % params.save_summary_steps == 0:
                # Perform a mini-batch update
                _, global_step_val, loss_val, labels_val, probabilities_val = sess.run([train_op, global_step, loss,
                                                                                        labels, probabilities])

                gap = eval_util.calculate_gap(probabilities_val, labels_val)

                logging.info("- Train metrics [in batch]: " + " GAP: " +
                             ("%.2f" % gap) + " Loss: " + str(loss_val))

                gap_summary = summary_pb2.Summary.Value(tag="gAP", simple_value=gap)
                loss_summary = summary_pb2.Summary.Value(tag="loss", simple_value=loss_val)
                summary = summary_pb2.Summary(value=[gap_summary, loss_summary])
                writer.add_summary(summary, global_step_val)
            else:
                sess.run(train_op)

        except tf.errors.OutOfRangeError:
            break


def train_and_evaluate(train_model_spec, dev_model_spec, model_dir, params, restore_from=None):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (dict) contains the graph operations or nodes needed for training
        eval_model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver() # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)
    begin_at_epoch = 0

    with tf.Session() as sess:
        # Initialize model variables
        sess.run(train_model_spec['variable_init_op'])

        # Reload weights from directory if specified
        if restore_from is not None:
            logging.info("Restoring parameters from {}".format(restore_from))
            if os.path.isdir(restore_from):
                restore_from = tf.train.latest_checkpoint(restore_from)
                begin_at_epoch = int(restore_from.split('-')[-1])
            last_saver.restore(sess, restore_from)

        # For tensorboard (takes care of writing summaries to files)
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summaries'), sess.graph)

        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, begin_at_epoch + params.num_epochs))
            # Compute number of batches in one epoch (one full pass over the training set)
            train_sess(sess, train_model_spec, train_writer, params)

            # Save weights
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step=epoch + 1)
