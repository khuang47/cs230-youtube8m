"""Tensorflow utility functions for training"""

import logging
import os
import tensorflow as tf
import metrics.eval_util as eval_util
from tensorflow.core.framework import summary_pb2


class SegmentTrainer:
    def __init__(self, train_model_spec, eval_model_spec, model_dir, params,
                 segment_restore_from, context_restore_from, context_aware_restore_from, context_ignore_restore_from):
        self.train_model_spec = train_model_spec
        self.eval_model_spec = eval_model_spec
        self.model_dir = model_dir
        self. params = params
        self.segment_restore_from = segment_restore_from
        self.context_restore_from = context_restore_from
        self.context_aware_restore_from = context_aware_restore_from
        self.context_ignore_restore_from = context_ignore_restore_from
        self.eval_best_map = 0  # best mean average precision on evaluation set after certain steps.

    def train_sess(self, sess, train_writer, eval_writer,
                   segment_saver, context_aware_saver, context_ignore_saver, epoch_val):
        """Train the model for one epoch. It will save checkpoints and evaluate the model
        based on dev set along the way.
        """
        # Get relevant graph operations or nodes needed for training
        probabilities = self.train_model_spec['probabilities']
        labels = self.train_model_spec['labels']
        loss = self.train_model_spec['loss']
        train_op = self.train_model_spec['train_op']

        global_step = tf.train.get_global_step()

        tf.train.start_queue_runners(sess=sess)

        global_step_val = 0

        while True:
            try:
                # Evaluate summaries for tensorboard only once in a while
                if global_step_val % self.params.save_summary_steps == 0:
                    # Perform a mini-batch update
                    (_,
                     global_step_val,
                     loss_val,
                     labels_val,
                     probabilities_val) = sess.run([train_op,
                                                    global_step,
                                                    loss,
                                                    labels,
                                                    probabilities])

                    mean_ap = eval_util.calculate_map(probabilities_val, labels_val, self.params.vocab_size,
                                                      top_k=self.params.vocab_size)

                    logging.info("- Train metrics: " + " mAP: " + ("%.2f" % mean_ap) + " Loss: " + str(loss_val) +
                                 " Epoch: " + ("%d" % epoch_val) + " Step: " + ("%d" % global_step_val))

                    train_map_summary = summary_pb2.Summary.Value(tag="mAP(training)", simple_value=mean_ap)
                    train_loss_summary = summary_pb2.Summary.Value(tag="loss(training)", simple_value=loss_val)
                    train_summary = summary_pb2.Summary(value=[train_map_summary, train_loss_summary])
                    train_writer.add_summary(train_summary, global_step_val)
                else:
                    sess.run(train_op)

                if global_step_val % self.params.evaluate_steps == 0:
                    # Evaluate on eval set
                    self.evaluate_sess(sess, global_step_val, eval_writer,
                                       segment_saver, context_aware_saver, context_ignore_saver)

            except tf.errors.OutOfRangeError:
                logging.info("out of range !!!!")
                break

    def evaluate_sess(self, sess, global_step_val, eval_writer,
                      segment_saver, context_aware_saver, context_ignore_saver):
        """Evaluation the model on evaluation set after certain steps.
        """
        probabilities = self.eval_model_spec['probabilities']
        labels = self.eval_model_spec['labels']
        loss = self.eval_model_spec['loss']

        # compute metrics over the dataset
        try:
            loss_val, labels_val, probabilities_val = sess.run([loss, labels, probabilities])
            curr_mean_ap = eval_util.calculate_map(probabilities_val, labels_val, self.params.vocab_size,
                                                   top_k=self.params.vocab_size)
            logging.info("- Evaluation metrics after " + ("%d" % global_step_val) + " steps: " + " mAP: " +
                         ("%.2f" % curr_mean_ap) + " Loss: " + str(loss_val))

            eval_map_summary = summary_pb2.Summary.Value(tag="mAP(evaluation)", simple_value=curr_mean_ap)
            eval_loss_summary = summary_pb2.Summary.Value(tag="loss(evaluation)", simple_value=loss_val)
            eval_summary = summary_pb2.Summary(value=[eval_map_summary, eval_loss_summary])
            eval_writer.add_summary(eval_summary, global_step_val)

            if curr_mean_ap > self.eval_best_map:
                self.eval_best_map = curr_mean_ap
                logging.info("Evaluation: Higher mAP found, save weights for segment model,"
                             "context aware model and context ignore model.")
                segment_weights_save_path = os.path.join(self.model_dir, 'segment_weights', 'after-step')
                segment_saver.save(sess, segment_weights_save_path, global_step=global_step_val)

                context_aware_weights_save_path = os.path.join(self.model_dir, 'context_aware_weights', 'after-step')
                context_aware_saver.save(sess, context_aware_weights_save_path, global_step=global_step_val)

                context_ignore_weights_save_path = os.path.join(self.model_dir, 'context_ignore_weights', 'after-step')
                context_ignore_saver.save(sess, context_ignore_weights_save_path, global_step=global_step_val)

        except tf.errors.OutOfRangeError:
            logging.info("Reach the end of eval dataset of segmentation model.")

    def train_and_evaluate(self):
        """Train the model and evaluate every epoch.
        """
        # Initialize tf.Saver instances to save weights during training
        context_varlist = {v.name[len("context_model/"):]: v
                           for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="context_model/")}
        context_model_best_saver = tf.train.Saver(var_list=context_varlist,
                                                  max_to_keep=1)

        context_aware_varlist = {v.name[len("context_aware_model/"):]: v
                                 for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="context_aware_model/")}
        context_aware_model_best_saver = tf.train.Saver(var_list=context_aware_varlist,
                                                        max_to_keep=1)

        context_ignore_varlist = {v.name[len("context_ignore_model/"):]: v
                                  for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="context_ignore_model/")}
        context_ignore_model_best_saver = tf.train.Saver(var_list=context_ignore_varlist,
                                                         max_to_keep=1)

        segment_varlist = {v.name: v
                           for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="segment_model/")}
        segment_model_best_saver = tf.train.Saver(var_list=segment_varlist,
                                                  max_to_keep=1)

        with tf.Session() as sess:
            # Initialize model variables
            sess.run(self.train_model_spec['variable_init_op'])

            sess.run(self.train_model_spec['iterator_init_op'])

            sess.run(self.eval_model_spec['iterator_init_op'])

            # Reload weights from directory if specified
            if (self.context_restore_from is None) or (
                    self.context_aware_restore_from is None) or (
                    self.context_ignore_restore_from is None):
                logging.error("Cannot find parameters to restore from!")
            else:
                logging.info("Restoring parameters for context model from {}".format(self.context_restore_from))
                logging.info("Restoring parameters for context aware model from {}"
                             .format(self.context_aware_restore_from))
                logging.info("Restoring parameters for context ignore model from {}".
                             format(self.context_ignore_restore_from))

                if os.path.isdir(self.context_restore_from):
                    context_restore_from = tf.train.latest_checkpoint(self.context_restore_from)
                if os.path.isdir(self.context_aware_restore_from):
                    context_aware_restore_from = tf.train.latest_checkpoint(self.context_aware_restore_from)
                if os.path.isdir(self.context_ignore_restore_from):
                    context_ignore_restore_from = tf.train.latest_checkpoint(self.context_ignore_restore_from)

                context_model_best_saver.restore(sess, context_restore_from)
                context_aware_model_best_saver.restore(sess, context_aware_restore_from)
                context_ignore_model_best_saver.restore(sess, context_ignore_restore_from)

            if self.segment_restore_from is not None:
                logging.info("Restoring parameters for segment model from {}".
                             format(self.segment_restore_from))
                if os.path.isdir(self.segment_restore_from):
                    segment_restore_from = tf.train.latest_checkpoint(self.segment_restore_from)

                segment_model_best_saver.restore(sess, segment_restore_from)

            # For tensorboard (takes care of writing summaries to files)
            train_writer = tf.summary.FileWriter(os.path.join(self.model_dir, 'train_summaries'), sess.graph)
            eval_writer = tf.summary.FileWriter(os.path.join(self.model_dir, 'eval_summaries'), sess.graph)

            for epoch in range(1, self.params.num_epochs + 1):
                # Run one epoch
                logging.info("Segment Prediction: Epoch {}/{}".format(epoch, self.params.num_epochs))
                self.train_sess(sess,
                                train_writer,
                                eval_writer,
                                segment_model_best_saver,
                                context_aware_model_best_saver,
                                context_ignore_model_best_saver,
                                epoch)
