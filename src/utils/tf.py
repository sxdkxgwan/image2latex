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
        self.encoded_img_s   = tf.shape(encoded_img)
        self.training        = training

        self._n_channels = 512
        self._dim_e      = 100
        self._dim_o      = 100

        self._state_is_tuple = state_is_tuple
        self._num_units      = num_units
        self._num_proj       = num_proj

        super(AttnCell, self).__init__(num_units)


    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._dim_o)


    @property
    def output_size(self):
        return self._num_proj


    def __call__(self, inputs, state):
        """
        Args:
            inputs: the embedding of the previous word
            state: tuple: (h, o) where h is the hidden state and o is the vector 
                used to make the prediction of the previous word
        """
        if self._state_is_tuple:
            h, o = state

        with tf.variable_scope(type(self).__name__):
                # 1. compute e
                W_img    = tf.get_variable("W_img", shape=(self._n_channels, self._dim_e),
                                            dtype=tf.float32)
                W_h      = tf.get_variable("h", shape=(self._num_units, self._dim_e),
                                            dtype=tf.float32)
                beta     = tf.get_variable("beta", shape=(self._dim_e, 1), dtype=tf.float32)

                img_flat = tf.reshape(self.encoded_img, shape=[-1, self._n_channels])
                e        = tf.matmul(tf.tanh(tf.matmul(img_flat, W_img) + tf.matmul(h, W_h)), beta)
                e        = tf.reshape(e, shape=[self.encoded_img_s[0], self.encoded_img_s[1]*self.encoded_img_s[2]])

                # 2. compute a
                a = tf.nn.softmax(e)
                a = tf.reshape(e, shape=self.encoded_img_s[:3])

                # 3. compute c
                c = tf.reduce_sum(a * self.encoded_img, axis=[1, 2])

                # 4. compute new h
                inputs = tf.concat([inputs, c], axis=-1)
                h, _   = super(AttnCell, self).__call__(inputs, h)

                # 5. compute o
                W_c = tf.get_variable("W_c", shape=(self._n_channels, self._dim_o), dtype=tf.float32)
                W_h = tf.get_variable("W_h", shape=(self._num_units, self._dim_o), dtype=tf.float32)
                o   = tf.tanh(tf.matmul(h, W_h) + tf.matmul(c, W_c))

                # 6. compute scores
                W_o    = tf.get_variable("W_o", shape=(self._dim_o, self._num_proj))
                scores = tf.matmul(o, W_o)
                
                new_state = LSTMStateTuple(h, o)
                new_out   = scores
                
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
