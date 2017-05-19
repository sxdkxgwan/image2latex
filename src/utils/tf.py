import tensorflow as tf
import collections
from tensorflow.contrib.rnn import GRUCell
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py


_StateTuple = collections.namedtuple("StateTuple", ("h", "o", "y"))


class StateTuple(_StateTuple):
  __slots__ = ()

  @property
  def dtype(self):
    (h, o, y) = self
    if not h.dtype == o.dtype == y.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s vs %s" %
                      (str(h.dtype), str(o.dtype), str(y.dtype)))
    return h.dtype


class AttnCell(GRUCell):
    def __init__(self, num_units, num_proj, encoded_img, training, E, state_is_tuple=True):
        """
        Args:
            num_units: (int) number of hidden units
            num_proj: (int) number of projection units
            encoded_img: (tf.Tensor) tensor of encoded image
            training: (tf.placeholder) bool
            E: (tf.Variable) embeddings matrix
            state_is_tuple: (bool)

        """
        self.encoded_img     = encoded_img
        self.encoded_img_s   = tf.shape(encoded_img)
        self.training        = training
        self.E               = E

        self._n_channels = 512
        self._dim_e      = 100
        self._dim_o      = 100
        self._dim_embed  = 100

        self._state_is_tuple = state_is_tuple
        self._num_units      = num_units
        self._num_proj       = num_proj

        super(AttnCell, self).__init__(num_units)


    @property
    def state_size(self):
        return StateTuple(self._num_units, self._dim_o, self._num_proj)


    @property
    def output_size(self):
        return self._num_proj


    def __call__(self, inputs, state):
        """
        Args:
            inputs: the embedding of the previous word for training only
            state: tuple: (h, o) where h is the hidden state and o is the vector 
                used to make the prediction of the previous word
        """
        if self._state_is_tuple:
            h, o, y = state

        with tf.variable_scope(type(self).__name__):
                # 1. compute e
                # TODO: check some existing code online to see if there is a more direct way 
                # of computing this - currently seems really slow but could be normal
                W_img    = tf.get_variable("W_img", shape=(self._n_channels, self._dim_e),
                                            dtype=tf.float32)
                W_h      = tf.get_variable("h", shape=(self._num_units, self._dim_e),
                                            dtype=tf.float32)
                beta     = tf.get_variable("beta", shape=(self._dim_e, 1), dtype=tf.float32)

                img_flat = tf.reshape(self.encoded_img, shape=[-1, self._n_channels])
                att_img  = tf.matmul(img_flat, W_img)
                shape    = [self.encoded_img_s[0], self.encoded_img_s[1], self.encoded_img_s[2], self._dim_e]
                att_img  = tf.reshape(att_img, shape=shape)
                att_h    = tf.matmul(h, W_h)
                att_h    = tf.expand_dims(att_h, axis=1)
                att_h    = tf.expand_dims(att_h, axis=1)
                s        = tf.tanh(att_img + att_h)
                s        = tf.reshape(s, shape=[-1, self._dim_e])
                e        = tf.matmul(s, beta)
                shape    = [self.encoded_img_s[0], self.encoded_img_s[1]*self.encoded_img_s[2]]
                e        = tf.reshape(e, shape=shape)

                # 2. compute a
                a = tf.nn.softmax(e)
                a = tf.reshape(e, shape=self.encoded_img_s[:3])
                a = tf.expand_dims(a, axis=-1)

                # 3. compute c
                c = tf.reduce_sum(a * self.encoded_img, axis=[1, 2])

                # 4. compute new h
                embedding_train = tf.nn.embedding_lookup(self.E, inputs)
                embedding_train = tf.reshape(embedding_train, shape=[-1, self._dim_embed])
                embedding_test  = tf.nn.embedding_lookup(self.E, tf.argmax(y, axis=-1))
                embedding       = tf.cond(self.training, lambda :embedding_train, lambda :embedding_test)
                inputs     = tf.concat([embedding, c], axis=-1)
                new_h, _   = super(AttnCell, self).__call__(inputs, h)

                # 5. compute o
                W_c = tf.get_variable("W_c", shape=(self._n_channels, self._dim_o), dtype=tf.float32)
                W_h = tf.get_variable("W_h", shape=(self._num_units, self._dim_o), dtype=tf.float32)
                new_o   = tf.tanh(tf.matmul(h, W_h) + tf.matmul(c, W_c))

                # 6. compute scores
                W_o    = tf.get_variable("W_o", shape=(self._dim_o, self._num_proj))
                new_y  = tf.matmul(o, W_o)
                
                new_state = StateTuple(new_h, new_o, new_y)
                
                return (new_y, new_state)


def conv2d(inputs, filters=64, kernel_size=3, strides=1, padding='SAME'):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides, padding)


def max_pooling2d(inputs, pool_size=2, strides=2, padding="SAME"):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding)


def batch_normalization(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training) 
