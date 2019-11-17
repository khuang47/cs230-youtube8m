"""Define the model."""

import math
import tensorflow as tf
import tensorflow.contrib.slim as slim


def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    rgb_audio = inputs['rgb_audio']

    rgb_audio = tf.layers.batch_normalization(rgb_audio, training=is_training, name="input_bn")

    rgb = rgb_audio[:, :, 0:1024]
    audio = rgb_audio[:, :, 1024:]
    with tf.variable_scope('rgb_vlad'):
        rgb_vlad = net_vlad(is_training, rgb, params, 1024)
    with tf.variable_scope('audio_vlad'):
        audio_vlad = net_vlad(is_training, audio, params, 128)
    vlad = tf.concat([rgb_vlad, audio_vlad], 1)

    with tf.variable_scope('fc'):
        out = tf.layers.dense(vlad, params.fc_size, kernel_initializer=tf.
                              random_normal_initializer(stddev=1 / math.sqrt(params.vlad_cluster_size)))
    with tf.variable_scope('gate'):
        gate = tf.layers.dense(out, params.fc_size, use_bias=False, kernel_initializer=tf
                               .random_normal_initializer(stddev=1 / math.sqrt(params.fc_size)))
    gate = tf.layers.batch_normalization(gate, training=is_training, name="gating_bn")

    gate = tf.sigmoid(gate)
    activation = tf.multiply(out, gate)

    # return mixture_of_expert(activation, params.num_mixtures, params)
    return MoE_with_gate(activation, is_training, params.num_mixtures, params)


def net_vlad(is_training, features, params, feature_size):
    """NetVLAD.

    Args:
        is_training: (bool) whether we are training or not
        features: (tensor) feature inputs. Either rgb(num_batch, num_frames, 1024) or audio(num_batch, num_frames, 128).
        params: (Params) hyper-parameters
        feature_size: feature size of each frame. rgb is 1024, audio is 128.
    Returns:
        vlad: Vector of Locally Aggregated Descriptors.
    """

    # 1x1 conv layer to convert features (num_batch, num_frames, feature_size) to clusters
    # (num_batch, num_frames, vlad_cluster_size).
    with tf.variable_scope('conv1x1'):
        aggregation = tf.layers.conv1d(features, params.vlad_cluster_size, 1,
                                       kernel_initializer=tf
                                       .random_normal_initializer(stddev=1 / math.sqrt(feature_size)))

    aggregation = tf.layers.batch_normalization(aggregation,
                                                momentum=params.bn_momentum,
                                                training=is_training,
                                                name="cluster_bn")
    aggregation = tf.nn.softmax(aggregation)

    prob_sum = tf.reduce_sum(aggregation, -2, keep_dims=True)
    cluster_weights = tf.get_variable("cluster_weights",
                                       [1, feature_size, params.vlad_cluster_size],
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    cluster_center = tf.multiply(prob_sum, cluster_weights)  # element-wise multiply
    aggregation = tf.transpose(aggregation, perm=[0, 2, 1])

    vlad = tf.matmul(aggregation, features)
    vlad = tf.transpose(vlad, perm=[0, 2, 1])
    vlad = tf.subtract(vlad, cluster_center)

    vlad = tf.nn.l2_normalize(vlad, 1)
    vlad = tf.reshape(vlad, [-1, params.vlad_cluster_size * feature_size])
    vlad = tf.nn.l2_normalize(vlad, 1)

    return vlad


def mixture_of_expert(features, num_mixtures, params):
    """A softmax over a mixture of logistic models (with L2 regularization).

    Args:
      features: (tensor) features after fully connected layer and context gating layer.
      num_mixtures: number of experts.
      params: (Params) hyper-parameters

    Returns:
      A tensor containing the probability predictions (batch_size x num_classes).
    """
    gate_activations = slim.fully_connected(
        features,
        params.vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(params.moe_l2),
        scope="gates")
    expert_activations = slim.fully_connected(
        features,
        params.vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(params.moe_l2),
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
                                     [-1, params.vocab_size])

    return final_probabilities


def MoE_with_gate(features, is_training, num_mixtures, params):
    gate_activations = slim.fully_connected(
        features,
        params.vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(params.moe_l2),
        scope="gates")

    expert_activations = slim.fully_connected(
        features,
        params.vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(params.moe_l2),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    probabilities = tf.reshape(probabilities_by_class_and_batch,
                               [-1, params.vocab_size])

    gating_weights = tf.get_variable("gating_prob_weights",
                                     [params.vocab_size, params.vocab_size],
                                     initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(params.vocab_size)))
    gates = tf.matmul(probabilities, gating_weights)

    gates = slim.batch_norm(
        gates,
        center=True,
        scale=True,
        is_training=is_training,
        scope="gating_prob_bn")
    gates = tf.sigmoid(gates)
    return tf.multiply(probabilities, gates)


def logistic_regression(features, params):
    """Creates a logistic regression.
    Args:
      features: (tensor) feature inputs. Either rgb(num_batch, num_frames, 1024) or audio(num_batch, num_frames, 128).
      params: (Params) hyper-parameters
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    probabilities = slim.fully_connected(
        features,
        params.vocab_size,
        activation_fn=tf.nn.sigmoid)
    return probabilities


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)
    # assert labels.get_shape().as_list() == [params.batch_size, params.vocab_size]

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        probabilities = build_model(is_training, inputs, params)


    #assert probabilities.get_shape().as_list() == [params.batch_size, params.vocab_size]

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
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, params.gradient_clip_norm)
        gradient_clip_op = optimizer.apply_gradients(zip(gradients, variables))

        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['loss'] = loss
    model_spec['labels'] = labels
    model_spec['probabilities'] = probabilities

    if is_training:
        model_spec['gradient_clip_op'] = gradient_clip_op
        model_spec['train_op'] = train_op
        model_spec['learning_rate'] = learning_rate_with_decay

    return model_spec


def calculate_loss(predictions, labels):
    epsilon = 10e-6

    float_labels = tf.cast(labels, tf.float32)
    cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
            1 - float_labels) * tf.log(1 - predictions + epsilon)
    cross_entropy_loss = tf.negative(cross_entropy_loss)
    return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


