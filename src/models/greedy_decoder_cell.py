import tensorflow as tf
import collections


class DecoderOutput(collections.namedtuple("DecoderOutput", 
                        ("logits", "ids"))):
    pass


class GreedyDecoderCell(object):
    def __init__(self, embeddings, attention_cell, batch_size, start_token):
        self._embeddings = embeddings
        self._attention_cell = attention_cell
        self._dim_embeddings = embeddings.shape[-1].value
        self._batch_size = batch_size
        self._start_token = start_token


    @property
    def output_dtype(self):
        """
        Needed for the custom dynamic_decode for the TensorArray of results
        """
        return DecoderOutput(
            logits=self._attention_cell.output_dtype,
            ids=tf.int32)


    def initial_state(self):
        """
        Return initial state for the lstm
        """
        return self._attention_cell.initial_state()


    def initial_inputs(self):
        """
        Returns initial inputs for the decoder (start token)
        """
        return tf.tile(tf.expand_dims(self._start_token, 0), 
                                multiples=[self._batch_size, 1])


    def step(self, time, state, embedding):
        # next step of attention cell
        logits, new_state = self._attention_cell.step(embedding, state)

        # get ids of words predicted and get embedding
        ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        new_embedding = tf.nn.embedding_lookup(self._embeddings, ids)

        # create new state of decoder
        new_output = DecoderOutput(logits, ids)
        
        return (new_output, new_state, new_embedding)


    def finalize(self, final_outputs, final_state):
        return final_outputs