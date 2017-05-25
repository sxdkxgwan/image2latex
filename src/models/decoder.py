import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from .attn_cell import TrainAttnCell, TestAttnCell
from utils.tf import batch_normalization

class Decoder(object):
    """
    Implements this paper https://arxiv.org/pdf/1609.04938.pdf
    """
    def __init__(self, config):
        self.config = config


    def __call__(self, training, encoded_img, formula, dropout, reuse=False):
        """
        Args:
            training: (tf.placeholder) bool
            encoded_img: (tf.Tensor) shape = (N, H, W, C)
            formula: (tf.placeholder), shape = (?, ?)
        Returns:
            pred: (tf.Tensor), shape = (?, max_length, vocab_size) scores of each class
        """
        # reshape image
        N    = tf.shape(encoded_img)[0]                           # batch size
        H, W = tf.shape(encoded_img)[1], tf.shape(encoded_img)[2] # image
        C    = encoded_img.shape[3].value                         # channels
        encoded_img_flat = tf.reshape(encoded_img, shape=[N, H*W, C])

        # get embeddings for training
        E = tf.get_variable("E", shape=[self.config.vocab_size, self.config.dim_embeddings], 
            dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        start_token = tf.get_variable("start_token", shape=[self.config.dim_embeddings],
            dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))

        embedding_formula = tf.nn.embedding_lookup(E, formula)
        start_token_      = tf.reshape(start_token, [1, 1, self.config.dim_embeddings])
        start_tokens      = tf.tile(start_token_, multiples=[N, 1, 1])
        embedding_train   = tf.concat([start_tokens, embedding_formula[:, :-1, :]], axis=1)
       
        # get Attention cell and formula for rnn
        train_attn_cell = TrainAttnCell(self.config.attn_cell_config, encoded_img_flat, 
                                        training, E, dropout=dropout)
        test_attn_cell  = TestAttnCell(self.config.attn_cell_config, encoded_img_flat, 
                                        training, E, dropout=1)

        # run attention cell
        with tf.variable_scope("attn_cell", reuse=False):
            train_outputs, _ = tf.nn.dynamic_rnn(train_attn_cell, embedding_train, 
                                    initial_state=train_attn_cell.initial_state())
            
        with tf.variable_scope("attn_cell", reuse=True):
            test_outputs, _  = tf.nn.dynamic_rnn(test_attn_cell, tf.expand_dims(formula, axis=-1),
                                    initial_state=test_attn_cell.initial_state(start_token))

        
        return train_outputs, test_outputs