import numpy as np
import tensorflow as tf
from utils.tf import conv2d, max_pooling2d, batch_normalization

class Encoder(object):
    def __init__(self, config):
        self.config = config


    def __call__(self, training, img):
        """
        Args:
            training: (tf.placeholder) tf.bool
            img: batch of img, shape = (?, height, width, channels)
        Returns:
            the encoded images, shape = (?, h', w', c')
        """        
        img = tf.cast(img, tf.float32) / 255.

        """
        ISSUE: seems to have performance issue (speed and accuracy) when
        using batch norm
        """

        with tf.variable_scope("convolutional_encoder"):
            out = conv2d(inputs=img, filters=64, kernel_size=3) 
            out = max_pooling2d(inputs=out)

            out = conv2d(inputs=out, filters=128, kernel_size=3)
            out = max_pooling2d(inputs=out)

            out = conv2d(inputs=out, filters=256, kernel_size=3)
            # out = batch_normalization(inputs=out, training=training) 

            out = conv2d(inputs=out, filters=256, kernel_size=3)
            out = max_pooling2d(inputs=out, pool_size=(2,1), strides=(2,1)) 

            out = conv2d(inputs=out, filters=512, kernel_size=3)
            # out = batch_normalization(inputs=out, training=training)
            out = max_pooling2d(inputs=out, pool_size=(1,2), strides=(1,2))

            out = conv2d(inputs=out, filters=512, kernel_size=3, padding='VALID')
            # out = batch_normalization(inputs=out, training=training) 

            return out
            



