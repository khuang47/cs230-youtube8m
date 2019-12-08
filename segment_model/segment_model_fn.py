import math
import tensorflow as tf
from model.model_fn import Youtube8mModel
import tensorflow.contrib.slim as slim


class SegmentModel:
    def __init__(self, is_training, embedding, params):
        self.is_training = is_training
        self.embedding = embedding
        self.params = params

    def forward(self):
        with tf.variable_scope('embedding_fc'):
            out = tf.layers.dense(self.embedding, self.params.embedding_fc_size, kernel_initializer=tf.
                                  random_normal_initializer(stddev=1 / math.sqrt(self.params.embedding_fc_size)))
            if self.params.use_batch_norm:
                out = tf.layers.batch_normalization(out,
                                                    momentum=self.params.bn_momentum,
                                                    training=self.is_training)
            out = tf.nn.relu(out)

        return self.mixture_of_expert(out, self.params.num_mixtures)

    def mixture_of_expert(self, features, num_mixtures):
        """A softmax over a mixture of logistic models (with L2 regularization).

        Args:
          features: (tensor) features after fully connected layer and context gating layer.
          num_mixtures: number of experts.

        Returns:
          A tensor containing the probability predictions (batch_size x num_classes).
        """
        gate_activations = slim.fully_connected(
            features,
            self.params.vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(self.params.moe_l2),
            scope="gates")
        expert_activations = slim.fully_connected(
            features,
            self.params.vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(self.params.moe_l2),
            scope="experts")

        gating_distribution = tf.nn.softmax(
            tf.reshape(
                gate_activations,
                [-1, num_mixtures + 1]))
        expert_distribution = tf.nn.sigmoid(
            tf.reshape(expert_activations,
                       [-1, num_mixtures]))

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, self.params.vocab_size])

        return final_probabilities


def model_fn(mode, inputs, params, reuse=False):
    model_spec = inputs

    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)
    # context model to encode the entire video, and will be kept frozen during training.
    with tf.variable_scope('context_model', reuse=reuse):
        context_model = Youtube8mModel(is_training, inputs['rgb_audio'], params, trainable=False, get_embedding=True)
        context_model.forward()
        context_embedding = context_model.frame_embedding

    # context aware model that encodes the segments.
    with tf.variable_scope('context_aware_model', reuse=reuse):
        context_aware_model = Youtube8mModel(is_training, inputs['segment_rgb_audio'], params, trainable=True,
                                             get_embedding=True)
        context_aware_model.forward()
        segment_embedding = context_aware_model.frame_embedding

    with tf.variable_scope('context_ignore_model', reuse=reuse):
        context_ignore_model = Youtube8mModel(is_training, inputs['segment_rgb_audio'], params, trainable=True,
                                              get_embedding=False)
        context_ignore_probabilities = context_ignore_model.forward()

    embedding = tf.concat([context_embedding, segment_embedding], 1)

    # the fully connected part of the context aware algorithm.
    with tf.variable_scope('segment_model', reuse=reuse):
        segment_model = SegmentModel(is_training, embedding, params)
        context_aware_probabilities = segment_model.forward()

    probabilities = (context_ignore_probabilities + context_aware_probabilities) / 2
    loss = calculate_loss(probabilities, labels)
    reg_loss = tf.add_n(tf.losses.get_regularization_losses())
    loss = loss + params.regularisation_lambda * reg_loss

    if is_training:
        global_step = tf.train.get_or_create_global_step()

        context_aware_var_list = tf.get_collection(tf.GraphKeys.VARIABLES, scope="context_aware_model/")
        context_ignore_var_list = tf.get_collection(tf.GraphKeys.VARIABLES, scope="context_ignore_model/")
        segment_var_list = tf.get_collection(tf.GraphKeys.VARIABLES, scope="segment_model/")

        context_aware_optimizer = tf.train.AdamOptimizer(params.segment_learning_rate / 2)
        context_ignore_optimizer = tf.train.AdamOptimizer(params.segment_learning_rate / 2)
        segment_optimizer = tf.train.AdamOptimizer(params.segment_learning_rate)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            context_aware_train_op = context_aware_optimizer.minimize(loss, var_list=context_aware_var_list)
            context_ignore_train_op = context_ignore_optimizer.minimize(loss, var_list=context_ignore_var_list)
            segment_train_op = segment_optimizer.minimize(loss, global_step=global_step,
                                                          var_list=segment_var_list)
            train_op = tf.group(context_aware_train_op, context_ignore_train_op, segment_train_op)

    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['loss'] = loss
    model_spec['labels'] = labels
    model_spec['probabilities'] = probabilities

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec


def calculate_loss(predictions, labels):
    """Compute logtis cross entropy loss.
    """
    epsilon = 10e-6

    float_labels = tf.cast(labels, tf.float32)
    cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
            1 - float_labels) * tf.log(1 - predictions + epsilon)
    cross_entropy_loss = tf.negative(cross_entropy_loss)
    return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))



