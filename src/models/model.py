import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils.general import Progbar, get_logger
from utils.data_utils import minibatches, pad_batch_images, \
    pad_batch_formulas, load_vocab
from encoder import Encoder
from decoder import Decoder
from utils.evaluate import evaluate, write_answers

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
            self.training: bool, shape = ()
        """
        # hyper params
        self.lr = tf.placeholder(tf.float32, shape=(),
            name='lr')
        self.dropout = tf.placeholder(tf.float32, shape=(),
            name='dropout')
        self.training = tf.placeholder(tf.bool, shape=(),
            name="training")


        # input of the graph
        self.img = tf.placeholder(tf.uint8, shape=(None, None, None, 1), 
            name='img')
        self.formula = tf.placeholder(tf.int32, shape=(None, None),
            name='formula')
        self.formula_length = tf.placeholder(tf.int32, shape=(None, ), 
            name='formula_length')
        
        # for tensorboard
        tf.summary.scalar("lr", self.lr) 


    def get_feed_dict(self, img, training, formula=None, lr=None, dropout=1):
        """
        Returns a dict
        """
        # pad batch
        img = pad_batch_images(img)

        fd = {
            self.img: img, 
            self.dropout: dropout, 
            self.training: training,
        }

        if formula is not None:
            formula, formula_length = pad_batch_formulas(formula, 
                    self.config.id_PAD, self.config.id_END)
            fd[self.formula] = formula
            fd[self.formula_length] = formula_length
        if lr is not None:
            fd[self.lr] = lr
            
        return fd


    def add_pred_op(self):
        """
        Defines self.pred
        """        
        encoded_img = self.encoder(self.training, self.img)
        train, test = self.decoder(self.training, encoded_img, self.formula, self.dropout)

        self.pred_train = train
        self.pred_test  = test


    def add_loss_op(self):
        """
        Defines self.loss
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_train, 
                                                                labels=self.formula)
        mask = tf.sequence_mask(self.formula_length)
        losses = tf.boolean_mask(losses, mask)
        self.loss = tf.reduce_mean(losses)
        
        # to compute perplexity
        self.ce_words = tf.reduce_sum(losses) # sum of cross entropy for each word
        self.n_words = tf.reduce_sum(self.formula_length) # number of words

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


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


    def run_epoch(self, sess, epoch, train_set, lr_schedule):
        """
        Performs an epoch of training
        """
        nbatches = (len(train_set) + self.config.batch_size - 1) / self.config.batch_size
        prog = Progbar(target=nbatches)
        for i, (img, formula) in enumerate(minibatches(train_set, self.config.batch_size)):
            # get feed dict
            fd = self.get_feed_dict(img, training=True, formula=formula, lr=lr_schedule.lr,
                                    dropout=self.config.dropout)
            # update step
            loss_eval, _, summary = sess.run([self.loss, self.train_op, self.merged], feed_dict=fd)
            self.file_writer.add_summary(summary, epoch*nbatches + i)

            # logging
            prog.update(i + 1, [("loss", loss_eval), ("lr", lr_schedule.lr)])

            # update learning rate
            lr_schedule.update(t=epoch*nbatches + i)


    def run_evaluate(self, sess, val_set, lr_schedule=None):
        """
        Performs an epoch of evaluation

        Args:
            val_set: Dataset instance
        Returns:
            bleu score: 
            exact match score: 
        """

        vocab = load_vocab(self.config.path_vocab)
        rev_vocab = {idx: word for word, idx in vocab.iteritems()}

        references, hypotheses = [], []
        n_words, ce_words = 0, 0 # for perplexity, sum of ce for all words + nb of words
        
        for img, formula in minibatches(val_set, self.config.batch_size):
            fd = self.get_feed_dict(img, training=False, formula=formula, dropout=1)
            ce_words_eval, n_words_eval, predictions = sess.run(
                    [self.ce_words, self.n_words, self.pred_test], feed_dict=fd)
            n_words += n_words_eval
            ce_words += ce_words_eval
            predictions = self.generate_answer(predictions)
            for form, pred in zip(formula, predictions):
                references.append([form])
                hypotheses.append(pred)

        scores = evaluate(references, hypotheses, rev_vocab, 
                            self.config.path_answers, self.config.id_END)

        ce_mean = ce_words / float(n_words)
        scores["perplexity"] = np.exp(ce_mean)

        if lr_schedule is not None:
            lr_schedule.update(score=scores["perplexity"])

        return scores


    def generate_answer(self, predictions):
        """
        Args:
            predictions: array of shape (batch_size, max_formula_length, vocab_size)
        Returns:
            formula by choosing one word of the vocab at each time step
        """
        return np.argmax(predictions, axis=-1)


    def add_summary(self, sess): 
        # tensorboard stuff
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output, sess.graph)


    def initialize_sess(self, sess, saver=None, dir_reload=None):
        """
        Initializes the variables in a session

        Args:
            sess: (tf.Session) instance
            saver: (tf.saver) optional, to reload weights from
            dir_reload: (string) path to directory with weights
        """
        if dir_reload is not None and saver is not None:
            # restoring weights
            print("Restoring parameters from {}".format(dir_reload))
            saver.restore(sess, dir_reload)
        else:
            # initialize with random initializer
            print("Initializing session with random weights")
            sess.run(self.init) # initialize variables
        # tensorboard
        self.add_summary(sess) 


    def train(self, train_set, val_set, lr_schedule):
        """
        Global train procedure
        """
        best_score = None
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.initialize_sess(sess, saver, self.config.dir_reload)
            self.evaluate_sess(sess, val_set, lr_schedule)

            for epoch in range(self.config.n_epochs):
                print("Epoch {}/{}".format(epoch+1, self.config.n_epochs))
                self.run_epoch(sess, epoch, train_set, lr_schedule)
                scores = self.evaluate_sess(sess, val_set, lr_schedule)

                # save weights if we have new best perplexity on eval
                if best_score is None or scores["perplexity"] < best_score:
                    saver.save(sess, self.config.model_output)
                    best_score = scores["perplexity"]


    def evaluate_sess(self, sess, val_set, lr_schedule=None):
        """
        Global eval procedure
        """
        # logging
        sys.stdout.write("\r- Evaluating...")
        sys.stdout.flush()
        
        # computing scores
        scores = self.run_evaluate(sess, val_set, lr_schedule)
        scores_to_print = ", ".join(["{} {:04.2f}".format(k, v) for k, v in scores.iteritems()])

        # logging
        sys.stdout.write("\r")
        sys.stdout.flush()
        self.logger.info("- Eval: {}".format(scores_to_print))

        return scores


    def evaluate(self, test_set, dir_reload):
        """
        Evaluate on test set and reloads weights from dir_reload

        Args:
            test_set: (Dataset)
            dir_reload: (string) path to directory with weights
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.initialize_sess(sess, saver, dir_reload)
            self.evaluate_sess(sess, test_set)

