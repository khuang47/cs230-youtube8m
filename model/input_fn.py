"""Create the input data pipeline using `tf.data`"""

import pandas as pd
import numpy as np
import tensorflow as tf
import utils.input_util as input_util


class Reader:
    def __init__(self, is_training, filenames, params, vocab_path="input/segment_vocabulary.csv"):
        self.is_training = is_training
        self.filenames = filenames
        self.params = params
        vocab = pd.read_csv(vocab_path)
        self.label_mapping = {
            label: index for label, index in zip(vocab["Index"], vocab.index)
        }

    def parse_fn(self, example_proto):
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
        vid_labels = list(tf.sparse_tensor_to_dense(contexts['labels']).numpy())

        vid_labels_encoded = set([
            self.label_mapping[x] for x in vid_labels if x in self.label_mapping
        ])
        labels_np = np.zeros(self.params.vocab_size, dtype=np.int64)
        labels_np[list(vid_labels_encoded)] = 1
        labels = tf.convert_to_tensor(labels_np, dtype=tf.int64)

        rgb = input_util.get_video_matrix(features['rgb'], 1024, self.params.max_frames)
        audio = input_util.get_video_matrix(features['audio'], 128, self.params.max_frames)

        # concat rgb and audio feature so that they can be (l2) normalized together.
        rgb_audio = tf.concat([rgb, audio], 1)

        return labels, rgb_audio

    def filter_fn(self, example_proto):
        context_features = {
            'labels': tf.VarLenFeature(tf.int64),
            'id': tf.FixedLenFeature([], dtype=tf.string)
        }

        contexts = tf.parse_single_example(
            example_proto,
            features=context_features)
        vid_labels = list(tf.sparse_tensor_to_dense(contexts['labels']).numpy())

        for x in vid_labels:
            if x in self.label_mapping:
                return True

        return False

    def tf_parse_fn(self, example_proto):
        labels, rgb_audio = tf.py_function(self.parse_fn, [example_proto], (tf.int64, tf.float32))
        labels.set_shape(self.params.vocab_size)
        rgb_audio.set_shape((self.params.max_frames, 1152))

        return labels, rgb_audio

    def tf_filter_fn(self, example_proto):
        tf_bool = tf.py_function(self.filter_fn, [example_proto], tf.bool)
        return tf_bool

    def input_fn(self):
        """Input function for the tfrecords dataset.
        """

        if self.is_training:
            dataset = (tf.data.TFRecordDataset(tf.data.Dataset.from_tensor_slices(self.filenames).shuffle(100))
                       .shuffle(256).filter(self.tf_filter_fn)
                       .map(self.tf_parse_fn, num_parallel_calls=4)
                       .batch(self.params.batch_size_train, drop_remainder=True)
                       .prefetch(1))
        else:
            dataset = (tf.data.TFRecordDataset(tf.data.Dataset.from_tensor_slices(self.filenames))
                       .filter(self.tf_filter_fn)
                       .map(self.tf_parse_fn, num_parallel_calls=4)
                       .batch(self.params.batch_size_eval, drop_remainder=True)
                       .prefetch(1))

        # Create reinitializable iterator from dataset
        iterator = dataset.make_initializable_iterator()

        labels, rgb_audio = iterator.get_next()

        rgb_audio = tf.nn.l2_normalize(rgb_audio, len(rgb_audio.get_shape()) - 1)

        iterator_init_op = iterator.initializer

        inputs = {'labels': labels, 'rgb_audio': rgb_audio,
                  'iterator_init_op': iterator_init_op}
        return inputs
