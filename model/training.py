"""Tensorflow utility functions for training"""

import logging
import os
import tensorflow as tf
import metrics.eval_util as eval_util
from tensorflow.core.framework import summary_pb2


class Trainer:
    def __init__(self, train_model_spec, eval_model_spec, model_dir, params, restore_from=None):
        self.train_model_spec = train_model_spec
        self.eval_model_spec = eval_model_spec
        self.model_dir = model_dir
        self. params = params
        self.restore_from = restore_from
        self.eval_best_gap = 0  # best global average precision on evaluation set after certain steps.

    def train_sess(self, sess, train_writer, eval_writer, saver, epoch_val):
        """Train the model for one epoch. It will save checkpoints and evaluate the model
        based on dev set along the way.

        Args:
            sess: (tf.Session) current session
            train_writer: (tf.summary.FileWriter) summary writer for training
            eval_writer: (tf.summary.FileWriter) summary writer for evaluation
            saver: (tf.train.Saver) saver to save weights/checkpoints
            epoch_val: epoch number
        """
        # Get relevant graph operations or nodes needed for training
        probabilities = self.train_model_spec['probabilities']
        labels = self.train_model_spec['labels']
        loss = self.train_model_spec['loss']
        gradient_clip_op = self.train_model_spec['gradient_clip_op']
        train_op = self.train_model_spec['train_op']

        global_step = tf.train.get_global_step()

        learning_rate = self.train_model_spec['learning_rate']

        sess.run(self.train_model_spec['iterator_init_op'])

        global_step_val = 0
        while True:
            try:
                # Evaluate summaries for tensorboard only once in a while
                if global_step_val % self.params.save_summary_steps == 0:
                    # Perform a mini-batch update
                    (_, _,
                     global_step_val,
                     loss_val,
                     labels_val,
                     probabilities_val,
                     learning_rate_val) = sess.run([gradient_clip_op, train_op,
                                                    global_step,
                                                    loss,
                                                    labels,
                                                    probabilities,
                                                    learning_rate])

                    gap = eval_util.calculate_gap(probabilities_val, labels_val)

                    logging.info("- Train metrics: " + " GAP: " +
                                 ("%.2f" % gap) + " Loss: " + str(loss_val) +
                                 " Learning rate: " + str(learning_rate_val) +
                                 " Epoch: " + ("%d" % epoch_val) + " Step: " + ("%d" % global_step_val))

                    train_gap_summary = summary_pb2.Summary.Value(tag="gAP(training)", simple_value=gap)
                    train_loss_summary = summary_pb2.Summary.Value(tag="loss(training)", simple_value=loss_val)
                    train_summary = summary_pb2.Summary(value=[train_gap_summary, train_loss_summary])
                    train_writer.add_summary(train_summary, global_step_val)
                else:
                    sess.run(gradient_clip_op, train_op)

                if global_step_val % self.params.evaluate_steps == 0:
                    # Evaluate on eval set
                    self.evaluate_sess(sess, global_step_val, eval_writer, saver)

            except tf.errors.OutOfRangeError:
                break

    def evaluate_sess(self, sess, global_step_val, eval_writer, saver):
        """Evaluation the model on evaluation set after certain steps.

        Args:
            sess: (tf.Session) current session
            global_step_val: global step number
            eval_writer: (tf.summary.FileWriter) summary writer for evaluation
            saver: (tf.train.Saver) saver to save weights/checkpoints.
                Save the weights after better precision found.
        """
        probabilities = self.eval_model_spec['probabilities']
        labels = self.eval_model_spec['labels']
        loss = self.eval_model_spec['loss']

        # Load the evaluation dataset into the pipeline and initialize the metrics init op
        sess.run(self.eval_model_spec['iterator_init_op'])

        # compute metrics over the dataset
        try:
            loss_val, labels_val, probabilities_val = sess.run([loss, labels, probabilities])
            curr_gap = eval_util.calculate_gap(probabilities_val, labels_val)
            logging.info("- Evaluation metrics after " + ("%d" % global_step_val) + " steps: " + " GAP: " +
                         ("%.2f" % curr_gap) + " Loss: " + str(loss_val))

            eval_gap_summary = summary_pb2.Summary.Value(tag="gAP(evaluation)", simple_value=curr_gap)
            eval_loss_summary = summary_pb2.Summary.Value(tag="loss(evaluation)", simple_value=loss_val)
            eval_summary = summary_pb2.Summary(value=[eval_gap_summary, eval_loss_summary])
            eval_writer.add_summary(eval_summary, global_step_val)

            if curr_gap > self.eval_best_gap:
                self.eval_best_gap = curr_gap
                logging.info("Evaluation: Higher gAP found, save weights.")
                save_path = os.path.join(self.model_dir, 'weights', 'after-step')
                saver.save(sess, save_path, global_step=global_step_val)

        except tf.errors.OutOfRangeError:
            logging.info("Reach the end of eval dataset.")

    def train_and_evaluate(self):
        """Train the model and evaluate every epoch.
        """
        # Initialize tf.Saver instances to save weights during training
        video_level_varlist = {v.name[len("video_level_model/"):]: v
                               for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="video_level_model/")}
        best_saver = tf.train.Saver(var_list=video_level_varlist,
                                    max_to_keep=1)

        with tf.Session() as sess:
            # Initialize model variables
            sess.run(self.train_model_spec['variable_init_op'])

            # Reload weights from directory if specified
            if self.restore_from is not None:
                logging.info("Restoring parameters from {}".format(self.restore_from))
                if os.path.isdir(self.restore_from):
                    restore_from = tf.train.latest_checkpoint(self.restore_from)

                best_saver.restore(sess, restore_from)

            # For tensorboard (takes care of writing summaries to files)
            train_writer = tf.summary.FileWriter(os.path.join(self.model_dir, 'train_summaries'), sess.graph)
            eval_writer = tf.summary.FileWriter(os.path.join(self.model_dir, 'eval_summaries'), sess.graph)

            for epoch in range(1, self.params.num_epochs + 1):
                # Run one epoch
                logging.info("Epoch {}/{}".format(epoch, self.params.num_epochs))
                # Compute number of batches in one epoch (one full pass over the training set)
                self.train_sess(sess, train_writer, eval_writer, best_saver, epoch)
