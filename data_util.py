import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
from ztq_pylib.zfile.ztq_sm import *


def _parse_data(image_filepath, single_label):

    single_label = tf.cast(single_label, tf.int64)
    file_contents = tf.io.read_file(image_filepath)
    image = tf.image.decode_bmp(file_contents, channels=3)

    return image, single_label



def load_dataset1(batch_size=32):


    file_root = '/tmp/a.txt'
    image_roots, labels = generate_fileroots_labels(file_root)


    dataset = tf.data.Dataset.from_tensor_slices((image_roots, labels))
    dataset = dataset.repeat(100).shuffle(buffer_size=2000) 
    dataset = dataset.map(_parse_data, num_parallel_calls=16)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


