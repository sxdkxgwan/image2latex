import sys
import numpy as np
import tensorflow as tf
import time
import tensorflow.contrib.layers as layers
from utils.general import Progbar
from utils.data_utils import minibatches, pad_batch_images, \
    pad_batch_formulas, load_vocab
from encoder import Encoder
from decoder import Decoder
from utils.evaluate import write_answers, evaluate


class Model(object):
    def __init__(self, config):
        # saveguard if previous model was defined
        tf.reset_default_graph()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)


    def build(self):
        """
        Builds model
        """
        self.config.logger.info("Building model...")
        self.add_placeholders_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.config.logger.info("- done.")


    def add_placeholders_op(self):
        """
        Add placeholder attributes
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
        img = pad_batch_images(img)

        fd = {
            self.img: img, 
            self.dropout: dropout, 
            self.training: training,
        }

        if formula is not None:
            formula, formula_length = pad_batch_formulas(formula, 
                    self.config.id_PAD, self.config.id_END)
            # print img.shape, formula.shape
            fd[self.formula] = formula
            fd[self.formula_length] = formula_length
        if lr is not None:
            fd[self.lr] = lr
            
        return fd


    def add_pred_op(self):
        """
        Defines self.pred
        """        
        encoded_img = self.encoder(self.training, self.img, self.dropout)
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

        # loss for training
        self.loss = tf.reduce_mean(losses)

        # # to compute perplexity for test
        self.ce_words = tf.reduce_sum(losses) # sum of CE for each word
        self.n_words = tf.reduce_sum(self.formula_length) # number of words
        
        # for tensorboard
        tf.summary.scalar("loss", self.loss)
        

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
            sess.run(tf.global_variables_initializer()) # initialize variables
        # tensorboard
        self.add_summary(sess) 


    def run_epoch(self, sess, epoch, train_set, lr_schedule):
        """
        Performs an epoch of training
        """
        # for logging
        tic = time.time()
        losses = 0 
        nbatches = (len(train_set) + self.config.batch_size - 1) / self.config.batch_size
        prog = Progbar(target=nbatches)
        # iterate over minibatches
        for i, (img, formula) in enumerate(minibatches(train_set, self.config.batch_size)):
            # get feed dict
            fd = self.get_feed_dict(img, training=True, formula=formula, lr=lr_schedule.lr,
                                    dropout=self.config.dropout)
            # update step
            loss_eval, _, summary = sess.run([self.loss, self.train_op, self.merged], feed_dict=fd)
            self.file_writer.add_summary(summary, epoch*nbatches + i)
            losses += loss_eval

            # logging
            prog.update(i + 1, 
                    values=[("loss", loss_eval), ("perplexity", np.exp(loss_eval))],
                    exact=[("lr", lr_schedule.lr)])

            # update learning rate
            lr_schedule.update(batch_no=epoch*nbatches + i)
        
        toc = time.time()
        self.config.logger.info("Epoch {} - time: {:04.2f}, loss: {:04.4f}, lr: {:04.5f}".format(
                        epoch, toc-tic, losses / float(max(i, 1)), lr_schedule.lr))


    def run_evaluate(self, sess, val_set, lr_schedule=None, path_results=None):
        """
        Performs an epoch of evaluation

        Args:
            sess: (tf.Session)
            val_set: Dataset instance
            lr_schedule: (instance of Lr schedule) optional
            path_results: (string) where to write the results
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
            ce_words_eval, n_words_eval, ids_eval = sess.run(
                    [self.ce_words, self.n_words, self.pred_test.ids], feed_dict=fd)

            if self.config.decoding == "greedy":
                ids_eval = np.expand_dims(ids_eval, axis=1)
                
            elif self.config.decoding == "beam_search":
                ids_eval = np.transpose(ids_eval, [0, 2, 1])

            n_words += n_words_eval
            ce_words += ce_words_eval
            for form, pred in zip(formula, ids_eval):
                # pred is of shape (number of hypotheses, time)
                references.append([form])
                hypotheses.append(pred)


        if path_results is None:
            path_results = self.config.path_results

        scores = evaluate(references, hypotheses, rev_vocab, 
                            path_results, self.config.id_END)

        ce_mean = ce_words / float(n_words)
        scores["perplexity"] = np.exp(ce_mean)

        if lr_schedule is not None:
            lr_schedule.update(score=scores["perplexity"])

        return scores


    def evaluate_sess(self, sess, val_set, lr_schedule=None, path_results=None):
        """
        Global eval procedure
        """
        # logging
        sys.stdout.write("\r- Evaluating...")
        sys.stdout.flush()
        
        # computing scores
        scores = self.run_evaluate(sess, val_set, lr_schedule, path_results)
        scores_to_print = ", ".join(["{} {:04.2f}".format(k, v) for k, v in scores.iteritems()])

        # logging
        sys.stdout.write("\r")
        sys.stdout.flush()
        self.config.logger.info("- Eval: {}".format(scores_to_print))

        return scores


    def evaluate(self, test_set, dir_reload, path_results):
        """
        Evaluate on test set and reloads weights from dir_reload.
        Writes a file with the predicted formulas

        Args:
            test_set: (Dataset)
            dir_reload: (string) path to directory with weights
        """
        # erase previous results from result path
        with open(path_results, "w") as f:
            pass
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.initialize_sess(sess, saver, dir_reload)
            # if we haven't written the results.txt file yet
            self.config.logger.info("Evaluation on Test set:")
            scores_sess = self.evaluate_sess(sess, test_set, path_results=path_results)

            return scores_sess


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
                    self.config.logger.info("- Found new best perplexity. Saving model.")
                    saver.save(sess, self.config.model_output)
