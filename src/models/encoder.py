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
        pass