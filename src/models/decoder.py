import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils.tf import TrainAttnCell, TestAttnCell


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
            encoded_img: (tf.Tensor) shape = (N, H, W, C)
            formula: (tf.placeholder), shape = (?, ?)
        Returns:
            pred: (tf.Tensor), shape = (?, max_length, vocab_size) scores of each class
        """
        # reshape image to shape = (N, H*W, C)
        N    = tf.shape(encoded_img)[0]
        H, W = tf.shape(encoded_img)[1], tf.shape(encoded_img)[2]
        C    = encoded_img.shape[3].value
        encoded_img_flat = tf.reshape(encoded_img, shape=[N, H*W, C])

        # get embeddings
        E = tf.get_variable("E", shape=[self.config.vocab_size, self.config.dim_embeddings], 
                dtype=tf.float32)
        embedding_train = tf.nn.embedding_lookup(E, formula)
       
        # get Attention cell and formula for rnn
        train_attn_cell = TrainAttnCell(100, self.config.vocab_size, encoded_img_flat, training, E)
        test_attn_cell  = TestAttnCell(100, self.config.vocab_size, encoded_img_flat, training, E)

        # run attention cell
        with tf.variable_scope("attn_cell", reuse=False):
            train_outputs, _ = tf.nn.dynamic_rnn(train_attn_cell, embedding_train, dtype=tf.float32)

        with tf.variable_scope("attn_cell", reuse=True):
            test_outputs, _  = tf.nn.dynamic_rnn(test_attn_cell, embedding_train, dtype=tf.float32)

        pred = tf.cond(training, lambda: train_outputs, lambda: test_outputs)
        
        return pred