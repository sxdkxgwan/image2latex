import tensorflow as tf
import collections
from tensorflow.python.util import nest


class BeamSearchDecoderCellState(collections.namedtuple("BeamSearchDecoderCellState", 
                       ("cell_state", "log_probs", "finished"))):
    """
    State of the Beam Search decoding

    cell_state: shape = structure of [batch_size, beam_size, ?] 
        cell state for all the hypotheses
    embedding: shape = [batch_size, beam_size, embedding_size]
        embeddings of the previous time step for each hypothesis
    log_probs: shape = [batch_size, beam_size]
        log_probs of the hypotheses
    finished: shape = [batch_size, beam_size]
        boolean to know if one beam hypothesis has reached token id_END
    """
    pass


class BeamSearchDecoderOutput(collections.namedtuple("BeamSearchDecoderOutput", 
                        ("logits", "ids", "parents"))):
    """
    Stores the logic for the beam search decoding

    logits: shape = [batch_size, beam_size, vocab_size]
        scores before softmax of the beam search hypotheses
    ids: shape = [batch_size, beam_size] 
        ids of the best words at this time step
    parents: shape = [batch_size, beam_size] 
        ids of the beam index from previous time step
    """
    pass


class BeamSearchFinalOutput(collections.namedtuple("BeamSearchFinalOutput",
    ("logits", "ids"))):
    pass


class BeamSearchDecoderCell(object):
    def __init__(self, embeddings, cell, batch_size, start_token, beam_size, id_end):
        self._embeddings = embeddings
        self._cell = cell
        self._dim_embeddings = embeddings.shape[-1].value
        self._batch_size = batch_size
        self._start_token = start_token
        self._beam_size  = beam_size
        self._id_end = id_end
        self._vocab_size = embeddings.shape[0].value


    @property
    def output_dtype(self):
        """
        Needed for the custom dynamic_decode for the TensorArray of results
        """
        return BeamSearchDecoderOutput(
            logits=self._cell.output_dtype,
            ids=tf.int32,
            parents=tf.int32)


    def initial_state(self):
        """
        Return initial state for the decoder
        """
        # cell initial state
        cell_state = self._cell.initial_state()
        cell_state = nest.map_structure(lambda t: tile_beam(t, self._beam_size), 
                                        cell_state)

        # prepare other initial states
        log_probs =  tf.zeros([self._batch_size, self._beam_size],
            dtype=self._cell.output_dtype)

        finished = tf.zeros([self._batch_size, self._beam_size], dtype=tf.bool)

        return BeamSearchDecoderCellState(
            cell_state,
            log_probs,
            finished)


    def initial_inputs(self):
        return tf.tile(tf.reshape(self._start_token, [1, 1, self._dim_embeddings]),
                multiples=[self._batch_size, self._beam_size, 1])


    def step(self, time, state, embedding):
        # merge batch and beam dimension before callling step of cell
        cell_state = nest.map_structure(merge_batch_beam, state.cell_state)
        embedding = merge_batch_beam(embedding)

        # compute new logits
        logits, new_cell_state = self._cell.step(embedding, cell_state)

        # split batch and beam dimension before beam search logic
        new_logits = split_batch_beam(logits, self._beam_size)
        new_cell_state = nest.map_structure(
            lambda t: split_batch_beam(t, self._beam_size),
                new_cell_state)

        # compute log probs of the step
        # shape = [batch_size, beam_size, vocab_size]
        step_log_probs = tf.nn.log_softmax(new_logits)
        # shape = [batch_size, beam_size, vocab_size]
        step_log_probs = mask_probs(step_log_probs, self._id_end, state.finished)
        # shape = [batch_size, beam_size, vocab_size]
        log_probs = tf.expand_dims(state.log_probs, axis=-1) + step_log_probs

        # compute the best beams
        log_probs_flat = tf.reshape(log_probs, [self._batch_size, -1])
        new_probs, indices = tf.nn.top_k(log_probs_flat, self._beam_size)

        # of shape [batch_size, beam_size]
        new_ids = indices % self._vocab_size
        new_parents = indices // self._vocab_size

        # get ids of words predicted and get embedding
        new_ids = tf.cast(tf.argmax(new_logits, axis=-1), tf.int32)
        new_embedding = tf.nn.embedding_lookup(self._embeddings, new_ids)

        # compute end of beam
        new_finished = tf.logical_or(state.finished, tf.equal(new_ids, self._id_end))

        # create new state of decoder
        new_state  = BeamSearchDecoderCellState(
            new_cell_state,
            new_probs,
            new_finished)
        
        new_output = BeamSearchDecoderOutput(
            new_logits, new_ids, new_parents)
        
        return (new_output, new_state, new_embedding)


    def finalize(self, final_outputs, final_state):
        """
        Args:
            final_outputs: structure of tensors of shape 
                    [T, batch_size, beam_size, ...]
            final_state: instance of BeamSearchDecoderOutput
        """
        ids = final_outputs.ids
        parents = final_outputs.parents
        logits = final_outputs.logits

        return BeamSearchFinalOutput(
            logits=logits[:, :, 0, :],
            ids=ids[:, :, 0])
        

def merge_batch_beam(t):
    """
    Args:
        t: tensor of shape [batch_size, beam_size, ...]
            whose dimensions after beam_size must be statically known

    Returns:
        t: tensorf of shape [batch_size * beam_size, ...]
    """
    if t.shape.ndims == 2:
        return tf.reshape(t, [-1, 1])
    elif t.shape.ndims == 3:
        return tf.reshape(t, [-1, t.shape[-1].value])
    elif t.shape.ndims == 4:
        return tf.reshape(t, [-1, t.shape[-2].value, t.shape[-1].value])
    else:
        raise NotImplementedError


def split_batch_beam(t, beam_size):
    """
    Args:
        t: tensorf of shape [batch_size*beam_size, ...]

    Returns:
        t: tensor of shape [batch_size, beam_size, ...]
    """
    if t.shape.ndims == 1:
        return tf.reshape(t, [-1, beam_size])
    elif t.shape.ndims == 2:
        return tf.reshape(t, [-1, beam_size, t.shape[-1].value])
    elif t.shape.ndims == 3:
        return tf.reshape(t, [-1, beam_size, t.shape[-2].value, t.shape[-1].value])
    else:
        raise NotImplementedError


def tile_beam(t, beam_size):
    """
    Args:
        t: tensor of shape [batch_size, ...]

    Returns:
        t: tensorf of shape [batch_size, beam_size, ...]
    """
    # shape = [batch_size, 1 , x]
    t = tf.expand_dims(t, axis=1)
    if t.shape.ndims == 2:
        multiples = [1, beam_size]
    elif t.shape.ndims == 3:
        multiples = [1, beam_size, 1]
    elif t.shape.ndims == 4:
        multiples = [1, beam_size, 1, 1]

    return tf.tile(t, multiples)


def mask_probs(probs, id_end, finished):
    """
    Args:
        probs: tensor of shape [batch_size, beam_size, vocab_size]
        id_end: (int)
        finished: tensor of shape [batch_size, beam_size], dtype = tf.bool
    """
    # one hot of shape [vocab_size]
    vocab_size = probs.shape[-1].value
    one_hot = tf.one_hot(id_end, vocab_size, dtype=probs.dtype)
    # expand dims of shape [batch_size, beam_size, 1]
    finished = tf.expand_dims(tf.cast(finished, probs.dtype), axis=-1)

    return (1. - finished) * probs + finished * one_hot
