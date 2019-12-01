"""Create the input data pipeline using `tf.data`"""

import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import utils.input_util as input_util


class SegmentReader:
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
            'id': tf.FixedLenFeature([], dtype=tf.string),
            "segment_labels": tf.io.VarLenFeature(tf.int64),
            "segment_start_times": tf.io.VarLenFeature(tf.int64),
            "segment_scores": tf.io.VarLenFeature(tf.float32)
        }
        sequence_features = {
            'rgb': tf.FixedLenSequenceFeature([], dtype=tf.string),
            'audio': tf.FixedLenSequenceFeature([], dtype=tf.string)
        }
        contexts, features = tf.parse_single_sequence_example(
            example_proto,
            context_features=context_features,
            sequence_features=sequence_features)

        segment_start_times = tf.sparse_tensor_to_dense(contexts['segment_start_times'])
        uniq_start_times, seg_idxs = tf.unique(segment_start_times,
                                               out_idx=tf.dtypes.int64)

        # the range of each segmentation is from (start time -3) to (end time + 3)
        range_mtx = tf.expand_dims(uniq_start_times, axis=-1) + tf.expand_dims(
            tf.range(-3, 8, dtype=tf.int64), axis=0)

        rgb = input_util.get_video_matrix(features['rgb'], 1024, self.params.max_frames)
        audio = input_util.get_video_matrix(features['audio'], 128, self.params.max_frames)

        # concat rgb and audio feature so that they can be (l2) normalized together.
        rgb_audio = tf.concat([rgb, audio], 1)

        # clip the range matrix in case it's out of bounds of the total frames.
        range_mtx = tf.clip_by_value(range_mtx, 0, rgb_audio.get_shape().as_list()[0])

        # Shape: [num_segment, segment_size, feature_dim].
        segment_rgb_audio = tf.gather_nd(rgb_audio,
                                         tf.expand_dims(range_mtx, axis=-1))

        # num_segment = tf.shape(segment_rgb_audio)[0]
        num_segment = segment_rgb_audio.get_shape().as_list()[0]

        # repeat video-level frame features num_segment times, to match with each segmented frame.
        rgb_audio_tiled = tf.tile(rgb_audio, [num_segment, 1])
        rgb_audio_tiled = tf.reshape(rgb_audio_tiled, [num_segment] + rgb_audio.get_shape().as_list())

        segment_labels = list(tf.sparse_tensor_to_dense(contexts['segment_labels']).numpy())

        for x in segment_labels:
            if x not in self.label_mapping:
                logging.error("ERROR: label not in segment vocab found:" + ("%d" % x))
        segment_labels_encoded_list = list([
            self.label_mapping[x] for x in segment_labels
        ])

        segment_labels_encoded = tf.convert_to_tensor(segment_labels_encoded_list, dtype=tf.int64)

        # Label indices for each segment, shape: [num_segment, 2]
        label_indices = tf.stack([seg_idxs, segment_labels_encoded],
                                 axis=-1)
        label_values = contexts["segment_scores"].values

        sparse_labels = tf.sparse.SparseTensor(label_indices, label_values,
                                               (num_segment, self.params.vocab_size))

        batch_labels = tf.sparse.to_dense(sparse_labels, validate_indices=False)

        return batch_labels, rgb_audio_tiled, segment_rgb_audio

    def tf_parse_fn(self, example_proto):
        labels, rgb_audio, segment_rgb_audio = tf.py_function(
            self.parse_fn, [example_proto], (tf.float32, tf.float32, tf.float32))
        labels.set_shape((None, self.params.vocab_size))
        rgb_audio.set_shape((None, self.params.max_frames, 1024 + 128))
        segment_rgb_audio.set_shape((None, 11, 1024 + 128))

        return labels, rgb_audio, segment_rgb_audio

    def input_fn(self):
        """Input function for the tfrecords dataset.
        """

        dataset = tf.data.TFRecordDataset(tf.data.Dataset.from_tensor_slices(self.filenames))

        # Create reinitializable iterator from dataset
        iterator = dataset.make_initializable_iterator()
        example = iterator.get_next()
        labels, rgb_audio, segment_rgb_audio = self.tf_parse_fn(example)
        iterator_init_op = iterator.initializer
        if self.is_training:
            batched_labels, batched_rgb_audio, batched_segment_rgb_audio = (
                tf.compat.v1.train.shuffle_batch([labels, rgb_audio, segment_rgb_audio],
                                                 batch_size=self.params.batch_size_train,
                                                 capacity=self.params.batch_size_train * 5,
                                                 min_after_dequeue=self.params.batch_size_train,
                                                 allow_smaller_final_batch=True, enqueue_many=True))
        else:
            batched_labels, batched_rgb_audio, batched_segment_rgb_audio = (
                tf.compat.v1.train.shuffle_batch([labels, rgb_audio, segment_rgb_audio],
                                                 batch_size=self.params.batch_size_eval,
                                                 capacity=self.params.batch_size_eval * 5,
                                                 min_after_dequeue=self.params.batch_size_eval,
                                                 allow_smaller_final_batch=True, enqueue_many=True))

        inputs = {'rgb_audio': batched_rgb_audio, 'labels': batched_labels,
                  'segment_rgb_audio': batched_segment_rgb_audio, 'iterator_init_op': iterator_init_op}

        return inputs
