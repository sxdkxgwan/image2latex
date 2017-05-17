import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMStateTuple
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py


class AttnCell(GRUCell):
    """
    GruAttn cell inspired by 224n ass4 presentation
    https://d1b10bmlvqabco.cloudfront.net/attach/iw9g8b9yxp46s8/iet5ntofwxf42m/j03ddaaw5kby/PA4Presentation.pdf
    """
    def __init__(self, num_units, num_proj, encoded_img, training, state_is_tuple=True):
        """
        Args:
            num_units: (int) number of hidden units
            encoder-output: (tf.Tensor) of shape (?, height, width, channels)
            training: (tf.placeholder) bool
            scope: (string)

        """
        self.encoded_img     = encoded_img
        self.training        = training

        self._state_is_tuple = state_is_tuple
        self._num_units      = num_units
        self._num_proj       = num_proj

        super(AttnCell, self).__init__(num_units)


    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_proj)


    def __call__(self, inputs, state):

        if self._state_is_tuple:
            h, o = state

        # gru_out, gru_state = super(AttnCell, self).__call__(inputs, state)
        with tf.variable_scope(type(self).__name__):
                # TODO compute e
                # TODO compute a
                # TODO compute c

                new_state = LSTMStateTuple(h, o)
                new_out   = h
                return (new_out, new_state)


def conv2d(inputs, filters=64, kernel_size=3, strides=1, padding='SAME'):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides, padding)


def max_pooling2d(inputs, pool_size=2, strides=2, padding="SAME"):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding)


def batch_normalization(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training) 


def embedding_lookup(ids, vocab_size, dim):
    E = tf.get_variable("E", dtype=tf.float32, shape=[vocab_size, dim])
    return tf.nn.embedding_lookup(E, ids)
