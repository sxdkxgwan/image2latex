import tensorflow as tf
import collections
from tensorflow.contrib.rnn import GRUCell, RNNCell, LSTMCell


"""
Define different sizes of states
"""

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


_StateTuple4 = collections.namedtuple("StateTuple", ("c", "h", "o", "v"))
class StateTuple4(_StateTuple4):
  __slots__ = ()

  @property
  def dtype(self):
    (c, h, o, v) = self
    if not c.dtype == h.dtype == o.dtype == v.dtype:
      raise TypeError("Inconsistent internal state: % vs %s vs %s vs %s" %
                      (str(c.dtype), str(h.dtype), str(o.dtype), str(v.dtype)))
    return h.dtype


class TrainAttnCell(RNNCell):
    def __init__(self, attn_cell_config, encoded_img_flat, training, E, dropout, reuse=False):
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
        self.dropout          = dropout
        self.E                = E

        # shapes of the input image
        self.N = tf.shape(self.encoded_img_flat)[0]   # batch size
        self.R = tf.shape(self.encoded_img_flat)[1]   # number of regions
        self.C = self.encoded_img_flat.shape[2].value # number of channels

        # hyperparameters
        self._dim_e          = attn_cell_config.get("dim_e", 512)
        self._dim_o          = attn_cell_config.get("dim_o", 512)
        self._num_units      = attn_cell_config.get("num_units", 512)
        self._num_proj       = attn_cell_config.get("num_proj", 512)
        self._dim_embeddings = attn_cell_config.get("dim_embeddings", 512)
        self._cell_type      = attn_cell_config.get("cell_type", "lstm").lower()
        self._reuse          = reuse
        
        # initializer internal cell
        if self._cell_type == "lstm":
            self.cell = LSTMCell(self._num_units, reuse=reuse)
            # (lstm memory, lstm hidden state, previous vector used for prediction) = (c, m, o)
            self._state_size = StateTuple3(self._num_units, self._num_units, self._dim_o)
        elif self._cell_type == "gru":
            self.cell = GRUCell(self._num_units, reuse=reuse)
            # (h, o)
            self._state_size = StateTuple2(self._num_units, self._dim_o)
        else:
            print("Unknown cell type provided.")
            raise NotImplementedError


    @property
    def state_size(self):
        return self._state_size


    @property
    def output_size(self):
        return self._num_proj


    def initial_state(self):
        """
        Return initial state for the lstm
        """
        # compute mean of image features
        C = self.encoded_img_flat.shape[-1].value
        img_0 = tf.reduce_mean(self.encoded_img_flat, axis=1)

        # first hidden state
        W_h_0 = tf.get_variable("W_h_0", shape=[C, self._num_units])
        b_h_0 = tf.get_variable("b_h_0", shape=[self._num_units])
        h_0 = tf.tanh(tf.matmul(img_0, W_h_0) + b_h_0)

        # first memory if lstm
        if self._cell_type == "lstm":
            W_c_0 = tf.get_variable("W_c_0", shape=[C, self._num_units])
            b_c_0 = tf.get_variable("b_c_0", shape=[self._num_units])
            c_0 = tf.tanh(tf.matmul(img_0, W_c_0) + b_c_0)

        # first o state
        W_o_0 = tf.get_variable("W_o_0", shape=[C, self._dim_o])
        b_o_0 = tf.get_variable("b_o_0", shape=[self._dim_o])
        o_0 = tf.tanh(tf.matmul(img_0, W_o_0) + b_o_0)

        # return first state
        if self._cell_type == "lstm":
            return StateTuple3(c_0, h_0, o_0)
        elif self._cell_type == "gru":
            return StateTuple2(h_0, o_0)


    def _compute_attention(self, img, h):
        """
        Computes attention

        Args:
            img: (batch_size, regions, channels) image representation
            h: (batch_size, num_units) hidden state 
        
        Returns:
            c: (batch_size, channels) context vector
        """
        # parameters
        att_W_img  = tf.get_variable("att_W_img", shape=(self.C, self._dim_e), dtype=tf.float32)
        att_W_h    = tf.get_variable("att_W_h", shape=(self._num_units, self._dim_e), dtype=tf.float32)
        att_beta   = tf.get_variable("att_beta", shape=(self._dim_e, 1), dtype=tf.float32)

        # compute attention over the image 
        _img    = tf.reshape(img, shape=[self.N*self.R, self.C])
        att_img = tf.matmul(_img, att_W_img)
        att_img = tf.reshape(att_img, shape=[self.N, self.R, self._dim_e])

        # compute attention over the hidden vector
        att_h = tf.matmul(h, att_W_h)
        att_h = tf.expand_dims(att_h, axis=1)

        # sum the two contributions
        s = tf.tanh(att_img + att_h)
        s = tf.reshape(s, shape=[self.N*self.R, self._dim_e])
        e = tf.matmul(s, att_beta)
        e = tf.reshape(e, shape=[self.N, self.R])

        # compute weights
        a = tf.nn.softmax(e)
        a = tf.expand_dims(a, axis=-1)

        # compute context
        c = tf.reduce_sum(a * img, axis=1)

        return c


    def _compute_h(self, embedding, o, prev_cell_state):
        x        = tf.concat([embedding, o], axis=-1)
        new_h, new_cell_state = self.cell.__call__(x, prev_cell_state)
        return new_cell_state, new_h


    def _step(self, embedding, prev_cell_state, o):
        """
        Args:
            embedding: shape =  (batch, dim_embeddings) embeddings
                from previous time step
            state: hidden state from previous time step
        """
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            # compute new h
            new_cell_state, new_h = self._compute_h(embedding, o, prev_cell_state)
            new_h = tf.nn.dropout(new_h, self.dropout) # apply dropout

            # compute attention
            c = self._compute_attention(self.encoded_img_flat, new_h)

            # compute o
            o_W_c   = tf.get_variable("o_W_c", shape=(self.C, self._dim_o), dtype=tf.float32)
            o_W_h   = tf.get_variable("o_W_h", shape=(self._num_units, self._dim_o), dtype=tf.float32)

            new_o = tf.tanh(tf.matmul(new_h, o_W_h) + tf.matmul(c, o_W_c))
            new_o = tf.nn.dropout(new_o, self.dropout) #apply dropout

            # new_o = new_h
            y_W_o  = tf.get_variable("y_W_o", shape=(self._dim_o, self._num_proj), dtype=tf.float32)
            new_y  = tf.matmul(new_o, y_W_o)

            return new_cell_state, new_o, new_y


    def __call__(self, inputs, state):
        """
        Args:
            inputs: the embedding of the previous word for training only
            state: tuple: (h, o) where h is the hidden state and o is the vector 
                used to make the prediction of the previous word
        """
        if self._cell_type == "lstm":
            c, m, o = state
            prev_cell_state = (c, m)
        elif self._cell_type == "gru":
            h, o = state
            prev_cell_state = h

        new_cell_state, new_o, new_y = self._step(inputs, prev_cell_state, o)

        if self._cell_type == "gru":
            new_state = StateTuple2(new_cell_state, new_o)
        elif self._cell_type == "lstm":
            c, m = new_cell_state
            new_state = StateTuple3(c, m, new_o)
        
        return (new_y, new_state)   


class TestAttnCell(TrainAttnCell):
    """
    Test-time version of the Attention cell. Must be placed
    in the same scope as the train_attn_cell with reuse=True 
    to share parameters.

    Main difference is that it doesn't receive an input embedding
    of the true formula, but just a dummy input. The embedding is the
    one from the word predicted at the previous time step.
    """
    @property
    def state_size(self):
        if self._cell_type == "lstm":
            # (c, m, o, embedding previous word)
            return StateTuple4(self._num_units, self._num_units, self._dim_o, self._dim_embeddings)
        elif self._cell_type == "gru":
            # (h, o, embedding previous word)
            return StateTuple3(self._num_units, self._dim_o, self._dim_embeddings)
        else:
            print("Unknown cell type provided.")
            raise NotImplementedError
        

    def initial_state(self, start_token):
        """
        Return initial state for the lstm
        """
        # tile start token embedding for the first time step
        N              = tf.shape(self.encoded_img_flat)[0]
        start_token_   = tf.reshape(start_token, [1, self._dim_embeddings])
        start_tokens   = tf.tile(start_token_, multiples=[N, 1])

        # initializer state of cell
        initial_state = super(TestAttnCell, self).initial_state()

        # return the cell state + embedding of previous word (start token here)
        if self._cell_type == "lstm":
            c, m, o = initial_state
            return StateTuple4(c, m, o, start_tokens)
        elif self._cell_type == "gru":
            h, o = initial_state
            return StateTuple3(h, o, start_tokens)


    def __call__(self, _, state):
        """
        Args:
            state: tuple: (h, o, v) where h is the hidden state and o is the vector 
                used to make the prediction of the previous word and v is the embedding
                of the previously predicted word
        """
        if self._cell_type == "lstm":
            c, m, o, v = state
            prev_cell_state = (c, m)
        elif self._cell_type == "gru":
            h, o, v = state
            prev_cell_state = h

        new_cell_state, new_o, new_y = self._step(v, prev_cell_state, o)

        idx    = tf.argmax(new_y, axis=-1)
        new_v  = tf.nn.embedding_lookup(self.E, idx)

        if self._cell_type == "gru":
            new_state = StateTuple3(new_cell_state, new_o, new_v)
        elif self._cell_type == "lstm":
            c, m = new_cell_state
            new_state = StateTuple4(c, m, new_o, new_v)
        
        return (new_y, new_state)