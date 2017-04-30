import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers



class Decoder(object):
    def __init__(self, config):
        self.config = config


    def __call__(self, img, formula):
        """
        Args:
            takes an encoded img
        Returns:
            decoded img
        """
        return tf.one_hot(formula, self.config.vocab_size)
        