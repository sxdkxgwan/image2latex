import tensorflow as tf


def conv2d(inputs, filters=64, kernel_size=3, strides=1, padding='SAME'):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides, padding)


def max_pooling2d(inputs, pool_size=2, strides=2, padding="SAME"):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding)


def batch_normalization(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training) 
