import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers



class Encoder(object):
    def __init__(self, config):
        self.config = config


    def __call__(self, img, is_training):
        """
        Args:
            img: batch of img
        Returns:
            the encoded images
        """
        img = tf.cast(img, tf.float32) / 255.

        with tf.variable_scope("convolutional_encoder"):
            # shape = (batch, imgH, imgW, 64)
            conv1 = layers.conv2d(inputs=img, num_outputs= 64, 
                kernel_size=3, stride=1, padding='SAME') 
            # shape = (batch, imgH/2, imgW/2, 64)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2,2), 
                strides=(2,2))

            # shape = (batch, imgH/2, imgW/2, 128)
            conv2 = layers.conv2d(inputs=pool1, num_outputs= 128, 
                kernel_size=3, stride=1, padding='SAME')
            # shape = (batch, imgH/4, imgW/4, 128)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2,2), 
                strides=(2,2)) 

            # shape = (batch, imgH/4, imgW/4, 256)
            conv3 = layers.conv2d(inputs=pool2, num_outputs= 256, 
                kernel_size=3, stride=1, padding='SAME')
            # shape = (batch, imgH/4, imgW/4, 256)
            norm3 = tf.layers.batch_normalization(inputs=conv3, axis=-1, 
                training=is_training) 

            # shape = (batch, imgH/4, imgW/4, 256)
            conv4 = layers.conv2d(inputs=norm3, num_outputs= 256, 
                kernel_size=3, stride=1, padding='SAME')
            # shape = (batch, imgH/8 , imgW/4, 256)
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=(2,1), 
                strides=(2,1)) 

            # shape = (batch, imgH/8 , imgW/4, 512)
            conv5 = layers.conv2d(inputs=pool4, num_outputs= 512, 
                kernel_size=3, stride=1, padding='SAME')
            # shape = (batch, imgH/8 , imgW/4, 512)
            norm5 = tf.layers.batch_normalization(inputs=conv5, axis=-1, 
                training=is_training)
            # shape = (batch, imgH/8 , imgW/8, 512)
            pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=(1,2), 
                strides=(1,2))
            # shape = (batch, imgH/8 -2 , imgW/8 - 2, 512)
            conv6 = layers.conv2d(inputs=pool5, num_outputs= 512, 
                kernel_size=3, stride=1, padding='VALID')
            # shape = (batch, imgH/8 -2 , imgW/8 - 2, 512)
            norm6 = tf.layers.batch_normalization(inputs=conv6, axis=-1, 
                training=is_training) 

            out_conv = norm6

            return out_conv
            



