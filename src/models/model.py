import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils.general import Progbar, get_logger
from utils.data_utils import minibatches, pad_batch_images, \
    pad_batch_formulas
from encoder import Encoder
from decoder import Decoder


class Model(object):
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(config.path_log)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)


    def build(self):
        """
        Builds model
        """
        self.logger.info("Building model...")
        self.add_placeholders_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()
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
            self.is_training: bool, shape = ()
        """
        # hyper params
        self.lr = tf.placeholder(tf.float32, shape=(),
            name='lr')
        self.dropout = tf.placeholder(tf.float32, shape=(),
            name='dropout')
        self.is_training = tf.placeholder(tf.bool, shape=(),
            name="is_training")


        # input of the graph
        self.img = tf.placeholder(tf.uint8, shape=(None, None, None, 1), 
            name='img')
        self.formula = tf.placeholder(tf.int32, shape=(None, None),
            name='formula')
        self.formula_length = tf.placeholder(tf.int32, shape=(None, ), 
            name='formula_length')
       

    def get_feed_dict(self, img, is_training, formula=None, lr=None, dropout=1):
        """
        Returns a dict
        """
        # pad batch
        img = pad_batch_images(img)

        fd = {
            self.img: img, 
            self.dropout: dropout, 
            self.is_training: is_training,
        }

        if formula is not None:
            formula, formula_length = pad_batch_formulas(formula)
            fd[self.formula] = formula
        if lr is not None:
            fd[self.lr] = lr
        if formula_length is not None:
            fd[self.formula_length] = formula_length

        return fd


    def add_pred_op(self):
        """
        Defines self.pred
        """        
        encoded_img = self.encoder(self.is_training, self.img)
        scores      = self.decoder(self.is_training, encoded_img, 
                    self.formula_length, formula=self.formula)

        self.pred   = scores


    def add_loss_op(self):
        """
        Defines self.loss
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, 
                                                                labels=self.formula)
        mask = tf.sequence_mask(self.formula_length)
        losses = tf.boolean_mask(losses, mask)
        self.loss = tf.reduce_mean(losses) + self.l2_loss() * 0.1


    def l2_loss(self):
        with tf.variable_scope("l2_loss"):
            variables = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in variables
                        if 'bias' not in v.name ])

        return l2_loss


    def add_train_op(self):
        """
        Defines self.train_op
        """
        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            # for batch norm beta gamma
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
            with tf.control_dependencies(update_ops): 
                self.train_op = optimizer.minimize(self.loss)


    def add_init_op(self):
        """
        Defines:
            self.init: op that initializes all variables
        """
        self.init = tf.global_variables_initializer()


    def run_epoch(self, sess, train_set):
        """
        Performs an epoch of training
        """
        nbatches = (len(train_set) + self.config.batch_size - 1) / self.config.batch_size
        prog = Progbar(target=nbatches)
        for i, (img, formula) in enumerate(minibatches(train_set, self.config.batch_size)):
            # get feed dict
            fd = self.get_feed_dict(img, is_training=True, formula=formula, lr=self.config.lr,
                                    dropout=self.config.dropout)
            # update step
            loss_eval, _ = sess.run([self.loss, self.train_op], feed_dict=fd)

            # logging
            prog.update(i + 1, [("loss", loss_eval)])


    def run_evaluate(self, val_set):
        """
        Performs an epoch of evaluation

        Args:
            val_set: Dataset instance
        Returns:
            bleu score: 
            exact match score: 
        """
        return 0, 0


    def train(self, train_set, val_set):
        """
        Global train procedure
        """
        with tf.Session() as sess:
            sess.run(self.init)

            for epoch in range(self.config.n_epochs):
                print("Epoch {}/{}".format(epoch+1, self.config.n_epochs))
                self.run_epoch(sess, train_set)
                self.evaluate(val_set)


    def evaluate(self, val_set):
        """
        Global eval procedure
        """
        sys.stdout.write("\r- Evaluating...")
        sys.stdout.flush()
        
        # do some stuff
        bleu, em = self.run_evaluate(val_set)

        sys.stdout.write("\r- Eval: BLEU {}, EM {}\n".format(bleu, em))
        sys.stdout.flush()
