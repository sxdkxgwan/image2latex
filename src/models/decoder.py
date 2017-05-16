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
        


    def __call__(self, img, formula):
        """
        Args:
            takes an encoded img
        Returns:
            decoded img
        """

        # img shape = (?, 18, 60, 512)
        # img_flat shape = (?, 552960)
        img_flat = layers.flatten(img)

        gru_attn_cell = GRUAttnCell(100, img_flat)
        
        # 1. run Gru Attn cell
        # fake inputs for the dynamic_rnn to loop over it a given number of times
        fake_inputs = tf.expand_dims(tf.zeros_like(formula, dtype=tf.float32), axis=-1)
        outputs, state = tf.nn.dynamic_rnn(gru_attn_cell, fake_inputs, dtype=tf.float32)

        # 2. predict distribution over next word

        # 3. regarder le truc d'harvard pour voir comment il font le decodage 
        #    si les images font pas la meme taille...
        return tf.one_hot(formula, self.config.vocab_size)
        