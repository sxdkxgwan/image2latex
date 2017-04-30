import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers



class Encoder(object):
    def __init__(self, config):
        self.config = config


    def __call__(self, img):
        """
        Args:
            img: batch of img
        Returns:
            the encoded images
        """
        with tf.variable_scope("encoder"):
            img = tf.cast(img, tf.float32) / 255.
            out = layers.conv2d(inputs=img, num_outputs=10, kernel_size=3, stride=1)