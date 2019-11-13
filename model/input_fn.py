"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import model.input_util as input_util


def parse_fn(example_proto):
    context_features = {
        'labels': tf.VarLenFeature(tf.int64),
        'id': tf.FixedLenFeature([], dtype=tf.string)
    }
    sequence_features = {
        'rgb': tf.FixedLenSequenceFeature([], dtype=tf.string),
        'audio': tf.FixedLenSequenceFeature([], dtype=tf.string)
    }
    contexts, features = tf.parse_single_sequence_example(
        example_proto,
        context_features=context_features,
        sequence_features=sequence_features)

    rgb = input_util.get_video_matrix(features['rgb'], 1024, 300)
    audio = input_util.get_video_matrix(features['audio'], 128, 300)
    labels = tf.reduce_sum(tf.one_hot(tf.sparse_tensor_to_dense(contexts['labels']), 3862), 0)

    # concat rgb and audio feature so that they can be (l2) normalized together.
    rgb_audio = tf.concat([rgb, audio], 1)

    return labels, rgb_audio


def input_fn(is_training, filenames, params):
    """Input function for the tfrecords dataset.

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    if is_training:
        dataset = (tf.data.TFRecordDataset(tf.data.Dataset.from_tensor_slices(filenames).shuffle(100))
                   .shuffle(256)
                   .map(parse_fn, num_parallel_calls=4)
                   .batch(params.batch_size, drop_remainder=True)
                   .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.TFRecordDataset(tf.data.Dataset.from_tensor_slices(filenames))
                   .map(parse_fn, num_parallel_calls=4)
                   .batch(params.batch_size, drop_remainder=True)
                   .prefetch(1)  # make sure you always have one batch ready to serve
        )
    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()

    labels, rgb_audio = iterator.get_next()

    rgb_audio = tf.nn.l2_normalize(rgb_audio, len(rgb_audio.get_shape()) - 1)

    iterator_init_op = iterator.initializer

    inputs = {'labels': labels, 'rgb_audio': rgb_audio,
              'iterator_init_op': iterator_init_op}
    return inputs
