import tensorflow as tf
import collections
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple


TrainAttnState = collections.namedtuple("TrainAttnState", ("cell_state", "o"))
TestAttnState  = collections.namedtuple("TestAttnState", ("cell_state", "o", "embedding"))


class TrainAttnCell(RNNCell):
    def __init__(self, cell, attn_mechanism, attn_cell_config, dropout):
        """
        Args:
            training: (tf.placeholder) bool
            E: (tf.Variable) embeddings matrix
        """
        # variables and tensors
        self.cell            = cell
        self._attn_mechanism = attn_mechanism
        self._dropout        = dropout

        # hyperparameters and shapes
        self._n_channels     = self._attn_mechanism._n_channels
        self._batch_size     = self._attn_mechanism._batch_size
        self._dim_e          = attn_cell_config.get("dim_e", 512)
        self._dim_o          = attn_cell_config.get("dim_o", 512)
        self._num_units      = attn_cell_config.get("num_units", 512)
        self._num_proj       = attn_cell_config.get("num_proj", 512)
        self._dim_embeddings = attn_cell_config.get("dim_embeddings", 512)
        
        # for RNNCell
        self._state_size = TrainAttnState(self.cell._state_size, self._dim_o)


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
        initial_cell_state = self._attn_mechanism.initial_cell_state(self.cell)
        initial_o          = self._attn_mechanism.initial_state("o", self._dim_o)

        return TrainAttnState(initial_cell_state, initial_o)


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
            x                     = tf.concat([embedding, o], axis=-1)
            new_h, new_cell_state = self.cell.__call__(x, prev_cell_state)
            new_h = tf.nn.dropout(new_h, self._dropout)

            # compute attention
            c = self._attn_mechanism.context(new_h)

            # compute o
            o_W_c   = tf.get_variable("o_W_c", shape=(self._n_channels, self._dim_o), dtype=tf.float32)
            o_W_h   = tf.get_variable("o_W_h", shape=(self._num_units, self._dim_o), dtype=tf.float32)

            new_o = tf.tanh(tf.matmul(new_h, o_W_h) + tf.matmul(c, o_W_c))
            new_o = tf.nn.dropout(new_o, self._dropout)

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
        prev_cell_state, o = state
        new_cell_state, new_o, new_y = self._step(inputs, prev_cell_state, o)
        new_state = TrainAttnState(new_cell_state, new_o)
        
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
    def __init__(self, E, cell, attn_mechanism, attn_cell_config, dropout):
        # embeddings
        self.E = E
        super(TestAttnCell, self).__init__(cell, attn_mechanism, attn_cell_config, dropout)


    @property
    def state_size(self):
        return TestAttnState(self.cell._state_size, self._dim_o, self._dim_embeddings)
        

    def initial_state(self, start_token):
        """
        Return initial state for the lstm
        """
        # tile start token embedding for the first time step
        start_token_   = tf.reshape(start_token, [1, self._dim_embeddings])
        start_tokens   = tf.tile(start_token_, multiples=[self._batch_size, 1])

        # initializer state of cell
        initial_state_cell, initial_o = super(TestAttnCell, self).initial_state()

        return TestAttnState(initial_state_cell, initial_o, start_tokens)


    def __call__(self, _, state):
        """
        Args:
            state: tuple: (h, o, v) where h is the hidden state and o is the vector 
                used to make the prediction of the previous word and v is the embedding
                of the previously predicted word
        """
        prev_cell_state, o, v = state

        new_cell_state, new_o, new_y = self._step(v, prev_cell_state, o)

        idx    = tf.argmax(new_y, axis=-1)
        new_v  = tf.nn.embedding_lookup(self.E, idx)
        new_state = TestAttnState(new_cell_state, new_o, new_v)
        
        return (new_y, new_state)