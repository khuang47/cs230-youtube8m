import tensorflow as tf


def resize_axis(tensor, axis, new_size, fill_value=0):
    """Truncates or pads a tensor to new_size on on a given axis.

    Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
    size increases, the padding will be performed at the end, using fill_value.

    Args:
      tensor: The tensor to be resized.
      axis: An integer representing the dimension to be sliced.
      new_size: An integer or 0d tensor representing the new value for
        tensor.shape[axis].
      fill_value: Value to use to fill any new entries in the tensor. Will be
        cast to the type of tensor.

    Returns:
      The resized tensor.
    """
    tensor = tf.convert_to_tensor(tensor)
    shape = tf.unstack(tf.shape(tensor))

    pad_shape = shape[:]
    pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

    shape[axis] = tf.minimum(shape[axis], new_size)
    shape = tf.stack(shape)

    resized = tf.concat([
        tf.slice(tensor, tf.zeros_like(shape), shape),
        tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
    ], axis)

    # Update shape.
    new_shape = tensor.get_shape().as_list()  # A copy is being made.
    new_shape[axis] = new_size
    resized.set_shape(new_shape)
    return resized


def dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """Dequantize the feature from the byte format to the float format.

    Args:
      feat_vector: the input 1-d vector.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      A float vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


def get_video_matrix(features,
                     feature_size,
                     max_frames,
                     max_quantized_value=2,
                     min_quantized_value=-2):
    """Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_frames: number of frames (rows) in the output feature_matrix
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      feature_matrix: matrix of all frame-features
    """
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    feature_matrix = dequantize(decoded_features,
                                max_quantized_value,
                                min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)
    return feature_matrix
