import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils.tf import GRUAttnCell


class Decoder(object):
    """
    Implements this paper https://arxiv.org/pdf/1609.04938.pdf
    """
    def __init__(self, config):
        self.config = config


    def get_embeddings(self, ids):
        embeddings = tf.get_variable("Embeddings", dtype=tf.float32,
            shape=[self.config.vocab_size, self.config.dim_embeddings])
        return tf.nn.embedding_lookup(embeddings, ids)
        

    def __call__(self, is_training, encoded_img, formula_length, formula=None):
        """
        Args:
            is_training: (tf.placeholder) bool
            encoded_img: (tf.Tensor) shape = (?, ?, ?, c')
            formula_length: (tf.placeholder), shape = (?)
            formula: (tf.placeholder), shape = (?, ?)
        Returns:
            pred: (tf.Tensor), shape = (?, max_length, vocab_size) scores of each class
        """

        # img shape = (?, 18, 60, 512)
        # img_flat shape = (?, 552960)
        gru_attn_cell = GRUAttnCell(100, encoded_img)
        
        # 1. run Gru Attn cell
        # fake inputs for the dynamic_rnn to loop over it a given number of times
        fake_inputs = tf.expand_dims(tf.zeros_like(formula, dtype=tf.float32), axis=-1)
        # outputs, state = tf.nn.dynamic_rnn(gru_attn_cell, fake_inputs, dtype=tf.float32)

        # 2. predict distribution over next word

        # 3. regarder le truc d'harvard pour voir comment il font le decodage 
        #    si les images font pas la meme taille...
        return tf.one_hot(formula, self.config.vocab_size)
        