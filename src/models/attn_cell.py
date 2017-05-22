import tensorflow as tf
import collections
from tensorflow.contrib.rnn import GRUCell, RNNCell


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


_StateTuple3 = collections.namedtuple("StateTuple", ("h", "o", "v"))
class StateTuple3(_StateTuple3):
  __slots__ = ()

  @property
  def dtype(self):
    (h, o, v) = self
    if not h.dtype == o.dtype == v.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s vs %s" %
                      (str(h.dtype), str(o.dtype), str(v.dtype)))
    return h.dtype



class TrainAttnCell(RNNCell):
    def __init__(self, attn_cell_config, encoded_img_flat, training, E):
        """
        Args:
            num_units: (int) number of hidden units
            num_proj: (int) number of projection units
            encoded_img_flat: (tf.Tensor) tensor of encoded image
                shape = (batch_size, nb of regions in image, channel size)
            training: (tf.placeholder) bool
            E: (tf.Variable) embeddings matrix
        """
        # variables and tensors
        self.encoded_img_flat = encoded_img_flat
        self.training         = training
        self.E                = E

        # shapes of the input image
        self.N = tf.shape(self.encoded_img_flat)[0]   # batch size
        self.R = tf.shape(self.encoded_img_flat)[1]   # number of regions
        self.C = self.encoded_img_flat.shape[2].value # number of channels

        # hyperparemeters
        self._dim_e          = attn_cell_config["dim_e"]
        self._dim_o          = attn_cell_config["dim_o"]
        self._num_units      = attn_cell_config["num_units"]
        self._num_proj       = attn_cell_config["num_proj"]
        self._dim_embeddings = attn_cell_config["dim_embeddings"]

        # regular cell init
        self.cell = GRUCell(self._num_units)


    @property
    def state_size(self):
        return StateTuple2(self._num_units, self._dim_o)


    @property
    def output_size(self):
        return self._num_proj


    def initial_state(self):
        """
        Return initial state for the lstm
        """
        C = self.encoded_img_flat.shape[-1].value
        img_0 = tf.reduce_mean(self.encoded_img_flat, axis=1)
        W_h_0 = tf.get_variable("W_h_0", shape=[C, self._num_units])
        b_h_0 = tf.get_variable("b_h_0", shape=[self._num_units])

        h_0 = tf.tanh(tf.matmul(img_0, W_h_0) + b_h_0)

        W_o_0 = tf.get_variable("W_o_0", shape=[C, self._dim_o])
        b_o_0 = tf.get_variable("b_o_0", shape=[self._dim_o])

        o_0 = tf.tanh(tf.matmul(img_0, W_o_0) + b_o_0)

        return StateTuple2(h_0, o_0)


    def _compute_attention(self, img, h, params):
        """
        Computes attention

        Args:
            img: (batch_size, regions, channels) image representation
            h: (batch_size, num_units) hidden state 
            params: (dict) params["W"] = tf.variable
        
        Returns:
            c: (batch_size, channels) context vector
        """
        # compute attention over the image 
        _img    = tf.reshape(img, shape=[self.N*self.R, self.C])
        att_img = tf.matmul(_img, params["W_img"])
        att_img = tf.reshape(att_img, shape=[self.N, self.R, self._dim_e])

        # compute attention over the hidden vector
        att_h = tf.matmul(h, params["W_h"])
        att_h = tf.expand_dims(att_h, axis=1)

        # sum the two contributions
        s = tf.tanh(att_img + att_h)
        s = tf.reshape(s, shape=[self.N*self.R, self._dim_e])
        e = tf.matmul(s, params["beta"])
        e = tf.reshape(e, shape=[self.N, self.R])

        # compute weights
        a = tf.nn.softmax(e)
        a = tf.expand_dims(a, axis=-1)

        # compute context
        c = tf.reduce_sum(a * img, axis=1)

        return c


    def _get_cell_params(self):
        """
        Returns tf variables for the __call__ method

        Returns:
            params: (dict) params["W"] = tf.Variable
        """
        params = dict()

        # to compute context vector (attention)
        params["W_img"] = tf.get_variable("W_img", shape=(self.C, self._dim_e), dtype=tf.float32)
        params["W_h"]   = tf.get_variable("h", shape=(self._num_units, self._dim_e), dtype=tf.float32)
        params["beta"]  = tf.get_variable("beta", shape=(self._dim_e, 1), dtype=tf.float32)

        # to compute new o (before the scores)
        params["W_c"]   = tf.get_variable("W_c", shape=(self.C, self._dim_o), dtype=tf.float32)
        params["W_h"]   = tf.get_variable("W_h", shape=(self._num_units, self._dim_o), dtype=tf.float32)

        # to compute new y (scores of next words)
        params["W_o"]   = tf.get_variable("W_o", shape=(self._dim_o, self._num_proj), dtype=tf.float32)

        return params


    def _compute_h(self, h, embedding, o):
        x        = tf.concat([embedding, o], axis=-1)
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
            # get params
            params = self._get_cell_params()
            
            # compute attention
            c = self._compute_attention(self.encoded_img_flat, h, params)

            # compute new h
            new_h = self._compute_h(h, inputs, o)

            # compute o
            new_o = tf.tanh(tf.matmul(new_h, params["W_h"]) + tf.matmul(c, params["W_c"]))

            # new state
            new_y  = tf.matmul(new_o, params["W_o"])
            new_state = StateTuple2(new_h, new_o)
            
            return (new_y, new_state)


class TestAttnCell(TrainAttnCell):
    """
    Test-time version of the Attention cell. Must be placed
    in the same scope name with reuse=True to share parameters

    Main difference is that it doesn't receive an input embedding
    of the true formula, but just a dummy input. The embedding is the
    one from the word predicted at the previous time step.

    """
    @property
    def state_size(self):
        return StateTuple3(self._num_units, self._dim_o, self._dim_embeddings)


    def initial_state(self, start_token):
        """
        Return initial state for the lstm
        """
        h_0, o_0 = super(TestAttnCell, self).initial_state()

        N              = tf.shape(self.encoded_img_flat)[0]
        start_token_   = tf.reshape(start_token, [1, self._dim_embeddings])
        start_tokens   = tf.tile(start_token_, multiples=[N, 1])

        return StateTuple3(h_0, o_0, start_tokens)


    def __call__(self, _, state):
        """
        Args:
            state: tuple: (h, o, v) where h is the hidden state and o is the vector 
                used to make the prediction of the previous word and v is the embedding
                of the previously predicted word
        """
        h, o, v = state
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            # get params
            params = self._get_cell_params()

            # compute attention
            c = self._compute_attention(self.encoded_img_flat, h, params)
            
            # compute new h
            new_h = self._compute_h(h, v, o)

            # compute o
            new_o = tf.tanh(tf.matmul(new_h, params["W_h"]) + tf.matmul(c, params["W_c"]))

            # new state
            new_y  = tf.matmul(new_o, params["W_o"])
            idx    = tf.argmax(new_y, axis=-1)
            new_v  = tf.nn.embedding_lookup(self.E, idx)

            new_state = StateTuple3(new_h, new_o, new_v)
            
            return (new_y, new_state)