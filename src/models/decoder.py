import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.rnn import GRUCell, LSTMCell
from utils.tf import batch_normalization, dynamic_decode
from .attention_mechanism import AttentionMechanism
from .attention_cell import AttentionCell
from .decoder_cell import DecoderCell


class Decoder(object):
    """
    Implements this paper https://arxiv.org/pdf/1609.04938.pdf
    """
    def __init__(self, config):
        self.config = config


    def __call__(self, training, encoded_img, formula, dropout):
        """
        Args:
            training: (tf.placeholder) bool
            encoded_img: (tf.Tensor) shape = (N, H, W, C)
            formula: (tf.placeholder), shape = (?, ?)
        Returns:
            pred: (tf.Tensor), shape = (?, max_length, vocab_size) scores of each class
        """
        # get embeddings for training
        E = tf.get_variable("E", shape=[self.config.vocab_size, self.config.dim_embeddings], 
            dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        start_token = tf.get_variable("start_token", shape=[self.config.dim_embeddings],
            dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))

        # embedding with start token
        batch_size        = tf.shape(formula)[0]
        embedding_formula = tf.nn.embedding_lookup(E, formula)
        start_token_      = tf.reshape(start_token, [1, 1, self.config.dim_embeddings])
        start_tokens      = tf.tile(start_token_, multiples=[batch_size, 1, 1])
        embedding_train   = tf.concat([start_tokens, embedding_formula[:, :-1, :]], axis=1) 

        # attention mechanism
        attention_mechanism = AttentionMechanism(encoded_img, self.config.attn_cell_config["dim_e"])

        # attention cell
        with tf.variable_scope("attn_cell", reuse=False):
            cell      = LSTMCell(self.config.attn_cell_config["num_units"])
            attn_cell = AttentionCell(cell, attention_mechanism, dropout, self.config.attn_cell_config)
            train_outputs, _ = tf.nn.dynamic_rnn(attn_cell, embedding_train, initial_state=attn_cell.initial_state())

        with tf.variable_scope("attn_cell", reuse=True):
            cell         = LSTMCell(self.config.attn_cell_config["num_units"], reuse=True)
            attn_cell    = AttentionCell(cell, attention_mechanism, dropout, self.config.attn_cell_config)
            decoder_cell = DecoderCell(E, attn_cell, start_token, batch_size)
            test_outputs, _ = dynamic_decode(decoder_cell, self.config.max_length_formula)
        
        return train_outputs, test_outputs