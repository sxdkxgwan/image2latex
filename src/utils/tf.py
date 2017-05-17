import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py


class GRUAttnCell(GRUCell):
    """
    GruAttn cell inspired by 224n ass4 presentation
    https://d1b10bmlvqabco.cloudfront.net/attach/iw9g8b9yxp46s8/iet5ntofwxf42m/j03ddaaw5kby/PA4Presentation.pdf
    """
    def __init__(self, num_units, encoder_output, scope=None):
        self.hs = encoder_output
        super(GRUAttnCell, self).__init__(num_units)


    def __call__(self, inputs, state, scope=None):
        gru_out, gru_state = super(GRUAttnCell, self).__call__(inputs, state, scope)
        with tf.variable_scope(scope or type(self).__name__):
                # TODO compute e
                # TODO compute a
                # TODO compute c
                out = gru_out

                return (out, out)


def conv2d(inputs, filters=64, kernel_size=3, strides=1, padding='SAME'):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides, padding)


def max_pooling2d(inputs, pool_size=2, strides=2, padding="SAME"):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding)


def batch_normalization(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training) 
