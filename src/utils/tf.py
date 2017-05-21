import tensorflow as tf
import collections
from tensorflow.contrib.rnn import GRUCell, RNNCell
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py


_StateTuple2 = collections.namedtuple("StateTuple", ("h", "o"))
class StateTuple2(_StateTuple2):
  __slots__ = ()

  @property
  def dtype(self):
    (h, o) = self
    if not h.dtype == o.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(h.dtype), str(o.dtype)))
    return h.dtype


_StateTuple3 = collections.namedtuple("StateTuple", ("h", "o", "y"))
class StateTuple3(_StateTuple3):
  __slots__ = ()

  @property
  def dtype(self):
    (h, o, y) = self
    if not h.dtype == o.dtype == y.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s vs %s" %
                      (str(h.dtype), str(o.dtype), str(y.dtype)))
    return h.dtype



class TrainAttnCell(RNNCell):
    def __init__(self, num_units, num_proj, encoded_img_flat, training, E):
        """
        Args:
            num_units: (int) number of hidden units
            num_proj: (int) number of projection units
            encoded_img_flat: (tf.Tensor) tensor of encoded image
                shape = (batch_size, nb of regions in image, channel size)
            training: (tf.placeholder) bool
            E: (tf.Variable) embeddings matrix
            state_is_tuple: (bool)

        """
        assert ((len(encoded_img_flat.shape.as_list()) == 3), 
                "encoded_img_flat must have 3 dimensions")

        # variables and tensors
        self.encoded_img_flat = encoded_img_flat
        self.training         = training
        self.E                = E

        # shapes of the input image
        self.N = tf.shape(self.encoded_img_flat)[0]   # batch size
        self.R = tf.shape(self.encoded_img_flat)[1]   # number of regions
        self.C = self.encoded_img_flat.shape[2].value # number of channels

        # hyperparemeters
        self._dim_e      = 100
        self._dim_o      = 100
        self._num_units  = num_units
        self._num_proj   = num_proj

        # regular cell init
        self.cell = GRUCell(num_units)


    @property
    def state_size(self):
        return StateTuple2(self._num_units, self._dim_o)


    @property
    def output_size(self):
        return self._num_proj


    def _compute_attention(self, img, h):
        """
        Computes attention

        Args:
            img: (batch_size, regions, channels) image representation
            h: (batch_size, num_units) hidden state 
        
        Returns:
            c: (batch_size, channels) context vector
        """
        # get variables
        W_img = tf.get_variable("W_img", shape=(self.C, self._dim_e), dtype=tf.float32)
        W_h   = tf.get_variable("h", shape=(self._num_units, self._dim_e), dtype=tf.float32)
        beta  = tf.get_variable("beta", shape=(self._dim_e, 1), dtype=tf.float32)

        # compute attention over the image 
        _img    = tf.reshape(img, shape=[self.N*self.R, self.C])
        att_img = tf.matmul(_img, W_img)
        att_img = tf.reshape(att_img, shape=[self.N, self.R, self._dim_e])

        # compute attention over the hidden vector
        att_h = tf.matmul(h, W_h)
        att_h = tf.expand_dims(att_h, axis=1)

        # sum the two contributions
        s = tf.tanh(att_img + att_h)
        s = tf.reshape(s, shape=[self.N*self.R, self._dim_e])
        e = tf.matmul(s, beta)
        e = tf.reshape(e, shape=[self.N, self.R])

        # compute weights
        a = tf.nn.softmax(e)
        a = tf.expand_dims(a, axis=-1)

        # compute context
        c = tf.reduce_sum(a * img, axis=1)

        return c


    def _compute_h(self, inputs, c, h, o, W_o):
        x        = tf.concat([inputs, c], axis=-1)
        new_h, _ = self.cell.__call__(x, h)
        return new_h


    def __call__(self, inputs, state):
        """
        Args:
            inputs: the embedding of the previous word for training only
            state: tuple: (h, o) where h is the hidden state and o is the vector 
                used to make the prediction of the previous word
        """
        h, o = state
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            # compute attention
            c = self._compute_attention(self.encoded_img_flat, h)

            # get some variables
            W_o   = tf.get_variable("W_o", shape=(self._dim_o, self._num_proj), dtype=tf.float32)
           
            # compute new h
            new_h = self._compute_h(inputs, c, h, o, W_o)

            # compute o
            W_c   = tf.get_variable("W_c", shape=(self.C, self._dim_o), dtype=tf.float32)
            W_h   = tf.get_variable("W_h", shape=(self._num_units, self._dim_o), dtype=tf.float32)
            new_o = tf.tanh(tf.matmul(h, W_h) + tf.matmul(c, W_c))

            # new states
            new_y  = tf.matmul(new_o, W_o)
            new_state = StateTuple2(new_h, new_o)
            
            return (new_y, new_state)


class TestAttnCell(TrainAttnCell):
    def _compute_h(self, inputs, c, h, o, W_o):
        y         = tf.matmul(o, W_o)
        embedding = tf.nn.embedding_lookup(self.E, tf.argmax(y, axis=-1))
        x         = tf.concat([embedding, c], axis=-1)
        new_h, _  = self.cell.__call__(x, h)
        return new_h



def conv2d(inputs, filters=64, kernel_size=3, strides=1, padding='SAME'):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides, padding)


def max_pooling2d(inputs, pool_size=2, strides=2, padding="SAME"):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding)


def batch_normalization(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training) 
