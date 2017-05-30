import tensorflow as tf
import collections


DecoderCellState = collections.namedtuple("DecoderCellState", 
                        ("attention_cell_state", "embedding"))


class DecoderCell(object):
    def __init__(self, embeddings, attention_cell, start_token, batch_size):
        self._embeddings = embeddings
        self._attention_cell = attention_cell
        self._dim_embeddings = embeddings.shape[-1].value
        self._start_token = start_token
        self._batch_size = batch_size
        

    def initial_state(self):
        """
        Return initial state for the lstm
        """
        # tile start token embedding for the first time step
        start_token    = self._start_token
        start_token_   = tf.reshape(start_token, [1, self._dim_embeddings])
        start_tokens   = tf.tile(start_token_, multiples=[self._batch_size, 1])

        attention_cell_initial_state = self._attention_cell.initial_state()

        return DecoderCellState(attention_cell_initial_state, start_tokens)


    def step(self, time, state):
        attention_cell_state, embedding = state

        new_attention_cell_state, new_y = self._attention_cell.step(embedding, attention_cell_state)

        idx           = tf.argmax(new_y, axis=-1)
        new_embedding = tf.nn.embedding_lookup(self._embeddings, idx)
        new_state     = DecoderCellState(new_attention_cell_state, new_embedding)
        
        return (new_y, new_state)