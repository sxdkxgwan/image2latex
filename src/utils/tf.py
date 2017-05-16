import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell


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
                pass