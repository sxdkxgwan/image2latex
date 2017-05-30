import tensorflow as tf


class AttentionMechanism(object):
    """
    Class to compute attention over an image
    """
    def __init__(self, img, dim_e):
        """
        Store the image under the right shape.

        We loose the H, W dimensions and merge them into a single 
        dimension that corresponds to "regions" of the image.

        Args:
            img: (tf.Tensor) image
            dim_e: (int) dimension of the intermediary vector used to 
                compute attention
        """

        """
        TODO: check that the last dimension of the image is not dynamic
        """
        # define self._img = (N, R, C)
        if len(img.shape) == 3:
            self._img = img
        elif len(img.shape) == 4:
            N    = tf.shape(img)[0]
            H, W = tf.shape(img)[1], tf.shape(img)[2] # image
            C    = img.shape[3].value                 # channels
            self._img = tf.reshape(img, shape=[N, H*W, C])
        else:
            print("Image shape not supported")
            raise NotImplementedError

        # mean of image (for initial states)
        self._img_mean = tf.reduce_mean(self._img, axis=1)

        # dimensions
        self._batch_size = tf.shape(self._img)[0]
        self._n_regions  = tf.shape(self._img)[1]
        self._n_channels = self._img.shape[2].value
        self._dim_e      = dim_e


    def context(self, h):
        """
        Computes attention

        Args:
            h: (batch_size, num_units) hidden state 
        
        Returns:
            c: (batch_size, channels) context vector
        """
        img = self._img

        # parameters
        dim_h      = h.shape[-1].value
        att_W_img  = tf.get_variable("att_W_img", shape=(self._n_channels, self._dim_e), dtype=tf.float32)
        att_W_h    = tf.get_variable("att_W_h", shape=(dim_h, self._dim_e), dtype=tf.float32)
        att_beta   = tf.get_variable("att_beta", shape=(self._dim_e, 1), dtype=tf.float32)

        # compute attention over the image 
        _img    = tf.reshape(img, shape=[self._batch_size*self._n_regions, self._n_channels])
        att_img = tf.matmul(_img, att_W_img)
        att_img = tf.reshape(att_img, shape=[self._batch_size, self._n_regions, self._dim_e])

        # compute attention over the hidden vector
        att_h = tf.matmul(h, att_W_h)
        att_h = tf.expand_dims(att_h, axis=1)

        # sum the two contributions
        s = tf.tanh(att_img + att_h)
        s = tf.reshape(s, shape=[self._batch_size*self._n_regions, self._dim_e])
        e = tf.matmul(s, att_beta)
        e = tf.reshape(e, shape=[self._batch_size, self._n_regions])

        # compute weights
        a = tf.nn.softmax(e)
        a = tf.expand_dims(a, axis=-1)

        # compute context
        c = tf.reduce_sum(a * img, axis=1)

        return c


    def initial_cell_state(self, cell):
        """
        Return initial state of a cell computed from the image
        Assumes cell.state_type is an instance of named_tuple.

        Ex: LSTMStateTuple

        Args:
            cell: (instance of RNNCell) must define _state_size
        """
        _states_0 = []
        for hidden_name in cell._state_size._fields:
            hidden_dim = getattr(cell._state_size, hidden_name)
            h = self.initial_state(hidden_name, hidden_dim)
            _states_0.append(h)

        initial_state_cell = type(cell.state_size)(*_states_0)

        return initial_state_cell


    def initial_state(self, name, dim):
        """
        Return initial state of dimension specified by dim
        Creates new variables given by name.
        """
        # initial state
        W = tf.get_variable("W_{}_0".format(name), shape=[self._n_channels, dim])
        b = tf.get_variable("b_{}_0".format(name), shape=[dim])
        h = tf.tanh(tf.matmul(self._img_mean, W) + b)

        return h