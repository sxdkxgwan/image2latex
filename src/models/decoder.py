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
        gru_attn_cell = GRUAttnCell(100, img)
        
        # 1. TODO: run cell
        # 2. predict distribution over next word
        # 3. regarder le truc d'harvard pour voir comment il font le decodage 
        #    si les images font pas la meme taille...
        return tf.one_hot(formula, self.config.vocab_size)
        