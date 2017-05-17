import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils.tf import AttnCell, embedding_lookup


class Decoder(object):
    """
    Implements this paper https://arxiv.org/pdf/1609.04938.pdf
    """
    def __init__(self, config):
        self.config = config


    def __call__(self, training, encoded_img, formula):
        """
        Args:
            training: (tf.placeholder) bool
            encoded_img: (tf.Tensor) shape = (?, ?, ?, c')
            formula: (tf.placeholder), shape = (?, ?)
        Returns:
            pred: (tf.Tensor), shape = (?, max_length, vocab_size) scores of each class
        """
        num_units = 100
        num_proj  = self.config.vocab_size
        gru_attn_cell = AttnCell(num_units, num_proj, encoded_img, training)
        formula_embed = embedding_lookup(formula, self.config.vocab_size, 
                                         self.config.dim_embeddings)

        # 1. run Gru Attn cell
        with tf.variable_scope("decoder"):
            # TODO: input_state?
            outputs, state = tf.nn.dynamic_rnn(gru_attn_cell, formula_embed, dtype=tf.float32)
            pred           = tf.one_hot(formula, self.config.vocab_size)
        
        return pred