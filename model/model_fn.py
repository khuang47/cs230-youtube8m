"""Define the model."""

import math
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Youtube8mModel:
    """the model

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters
        trainable: (bool)
    """
    def __init__(self, is_training, rgb_audio_features, params, trianable, get_embedding=False):
        self.is_training = is_training
        self.rgb_audio_features = rgb_audio_features
        self.params = params
        self.trianable = trianable
        self.frame_embedding = None
        self.get_embedding = get_embedding

    def forward(self):
        """ forward propagation to compute logits of the model (output distribution)

        Returns:
            output: (tf.Tensor) output of the model
        """

        rgb_audio = tf.layers.batch_normalization(self.rgb_audio_features,
                                                  training=self.is_training,
                                                  name="input_bn",
                                                  trainable=self.trianable)

        rgb = rgb_audio[:, :, 0:1024]
        audio = rgb_audio[:, :, 1024:]
        with tf.variable_scope('rgb_vlad'):
            rgb_vlad = self.net_vlad(rgb, 1024)
        with tf.variable_scope('audio_vlad'):
            audio_vlad = self.net_vlad(audio, 128)
        vlad = tf.concat([rgb_vlad, audio_vlad], 1)

        with tf.variable_scope('fc'):
            out = tf.layers.dense(vlad, self.params.fc_size, kernel_initializer=tf.
                                  random_normal_initializer(stddev=1 / math.sqrt(self.params.vlad_cluster_size)),
                                  trainable=self.trianable
                                  )
        with tf.variable_scope('gate'):
            gate = tf.layers.dense(out, self.params.fc_size, use_bias=False, kernel_initializer=tf
                                   .random_normal_initializer(stddev=1 / math.sqrt(self.params.fc_size)),
                                   trainable=self.trianable
                                   )
            gate = tf.layers.batch_normalization(gate, training=self.is_training, name="gating_bn",
                                                 trainable=self.trianable)

        gate = tf.sigmoid(gate)
        activation = tf.multiply(out, gate)

        self.frame_embedding = activation

        if not self.get_embedding:
            return self.moe_with_gate(activation, self.params.num_mixtures)

    def net_vlad(self, features, feature_size):
        """NetVLAD.

        Args:
            features: (tensor) feature inputs. Either rgb(num_batch, num_frames, 1024) or
                audio(num_batch, num_frames, 128).
            feature_size: feature size of each frame. rgb is 1024, audio is 128.
        Returns:
            vlad: Vector of Locally Aggregated Descriptors.
        """

        # 1x1 conv layer to convert features (num_batch, num_frames, feature_size) to clusters
        # (num_batch, num_frames, vlad_cluster_size).
        with tf.variable_scope('conv1x1'):
            aggregation = tf.layers.conv1d(features, self.params.vlad_cluster_size, 1,
                                           use_bias=False,
                                           kernel_initializer=tf
                                           .random_normal_initializer(stddev=1 / math.sqrt(feature_size)),
                                           trainable=self.trianable)

        aggregation = tf.layers.batch_normalization(aggregation,
                                                    momentum=self.params.bn_momentum,
                                                    training=self.is_training,
                                                    name="cluster_bn",
                                                    trainable=self.trianable)
        aggregation = tf.nn.softmax(aggregation)

        prob_sum = tf.reduce_sum(aggregation, -2, keep_dims=True)
        cluster_weights = tf.get_variable("cluster_weights",
                                          [1, feature_size, self.params.vlad_cluster_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)),
                                          trainable=self.trianable)
        cluster_center = tf.multiply(prob_sum, cluster_weights)  # element-wise multiply with broadcasting
        aggregation = tf.transpose(aggregation, perm=[0, 2, 1])

        vlad = tf.matmul(aggregation, features)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, cluster_center)

        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = tf.reshape(vlad, [-1, self.params.vlad_cluster_size * feature_size])
        vlad = tf.nn.l2_normalize(vlad, 1)

        return vlad

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
            scope="gates",
            trainable=self.trianable
        )
        expert_activations = slim.fully_connected(
            features,
            self.params.vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(self.params.moe_l2),
            scope="experts",
            trainable=self.trianable
        )

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

    def moe_with_gate(self, features, num_mixtures):
        """Creates a MoE classifier with gate.
        Args:
          features: (tensor) feature inputs. Either rgb(num_batch, num_frames, 1024)
            or audio(num_batch, num_frames, 128).
          num_mixtures: number of experts for this layer.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        gate_activations = slim.fully_connected(
            features,
            self.params.vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(self.params.moe_l2),
            scope="gates",
            trainable=self.trianable
        )

        expert_activations = slim.fully_connected(
            features,
            self.params.vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(self.params.moe_l2),
            scope="experts",
            trainable=self.trianable
        )

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        probabilities = tf.reshape(probabilities_by_class_and_batch,
                                   [-1, self.params.vocab_size])

        gating_weights = tf.get_variable("gating_prob_weights",
                                         [self.params.vocab_size, self.params.vocab_size],
                                         initializer=tf
                                         .random_normal_initializer(stddev=1 / math.sqrt(self.params.vocab_size)),
                                         trainable=self.trianable
                                         )
        gates = tf.matmul(probabilities, gating_weights)

        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=self.is_training,
            scope="gating_prob_bn",
            trainable=self.trianable
        )
        gates = tf.sigmoid(gates)
        return tf.multiply(probabilities, gates)

    def logistic_regression(self, features):
        """Creates a logistic regression classifier.
        Args:
          features: (tensor) feature inputs. Either rgb(num_batch, num_frames, 1024)
            or audio(num_batch, num_frames, 128).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        probabilities = slim.fully_connected(
            features,
            self.params.vocab_size,
            activation_fn=tf.nn.sigmoid,
            trainable=self.trianable
        )
        return probabilities


def model_fn(mode, inputs, params, trainable=True, reuse=False):
    """Model function defining the graph operations.

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # MODEL: define the layers of the model
    with tf.variable_scope('video_level_model', reuse=reuse):
        model = Youtube8mModel(is_training, inputs['rgb_audio'], params, trainable)
        probabilities = model.forward()

    loss = calculate_loss(probabilities, labels)
    reg_loss = tf.add_n(tf.losses.get_regularization_losses())
    loss = loss + params.regularisation_lambda * reg_loss

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        global_step = tf.train.get_or_create_global_step()
        learning_rate_with_decay = tf.train.exponential_decay(
            params.learning_rate,
            global_step * params.batch_size_train,
            params.learning_rate_decay_steps,
            params.learning_rate_decay,
            staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate_with_decay)

        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)

    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['loss'] = loss
    model_spec['labels'] = labels
    model_spec['probabilities'] = probabilities

    if is_training:
        model_spec['train_op'] = train_op
        model_spec['learning_rate'] = learning_rate_with_decay

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


