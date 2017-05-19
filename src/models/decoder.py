import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils.tf import AttnCell


class Decoder(object):
    """
    Implements this paper https://arxiv.org/pdf/1609.04938.pdf
    """
    def __init__(self, config):
        self.config = config


    def __call__(self, training, encoded_img, formula, reuse=False):
        """
        Args:
            training: (tf.placeholder) bool
            encoded_img: (tf.Tensor) shape = (?, ?, ?, c')
            formula: (tf.placeholder), shape = (?, ?)
        Returns:
            pred: (tf.Tensor), shape = (?, max_length, vocab_size) scores of each class
        """
        num_units = 100
        E = tf.get_variable("E", shape=[self.config.vocab_size, self.config.dim_embeddings], 
            dtype=tf.float32)
        gru_attn_cell   = AttnCell(num_units, self.config.vocab_size, encoded_img, training, E)
        formula_for_rnn = tf.expand_dims(formula, axis=-1)

        # 1. run Gru Attn cell
        with tf.variable_scope("decoder", reuse=reuse):
            # TODO: input_state?
            outputs, state = tf.nn.dynamic_rnn(gru_attn_cell, formula_for_rnn, dtype=tf.float32)
            # pred           = tf.one_hot(formula, self.config.vocab_size)
            pred = outputs

        
        return pred