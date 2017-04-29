import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils.general import Progbar, get_logger


class Model(object):
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(config.path_log)


    def build(self):
        """
        Builds model
        """
        self.logger.info("Building model...")
        pass
        self.logger.info("- done.")


    def add_placeholders_op(self):
        """
        Add placeholder attributes

        Defines:
            self.lr: float32, shape = ()
            self.img: uint8, shape = (None, None, None)
            self.formula: int32, shape = (None, None)
            self.formula_length: int32, shape = (None, )
            self.dropout: float32, shape = ()
        """
        pass


    def get_feed_dict(self, lr, img, formula, formula_length, dropout):
        """
        Returns a dict
        """
        pass


    def add_pred_op(self):
        """
        Defines self.pred
        """
        pass


    def add_loss_op(self):
        """
        Defines self.loss
        """
        pass


    def add_train_op(self):
        """
        Defines self.train_op
        """
        pass


    def add_init_op(self):
        """
        Defines:
            self.init: op that initializes all variables
        """
        pass


    def run_epoch(self, x_batch, y_batch):
        """
        Performs an epoch of training
        """
        pass


    def run_evaluate(self, x_batch, y_batch):
        """
        Performs an epoch of evaluation
        """
        pass


    def train(self):
        """
        Global train procedure
        """
        pass


    def evaluate(self):
        """
        Global eval procedure
        """
        pass
