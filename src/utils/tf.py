import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn


def conv2d(inputs, filters=64, kernel_size=3, strides=1, padding='SAME'):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides, padding, activation=tf.nn.relu)


def max_pooling2d(inputs, pool_size=2, strides=2, padding="SAME"):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding)


def batch_normalization(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training, axis=-1) 


def transpose_batch_time(t):
    if t.shape.ndims == 2:
        return tf.transpose(t, [1, 0])
    elif t.shape.ndims == 3:
        return tf.transpose(t, [1, 0, 2])
    else:
        raise NotImplementedError


def dynamic_decode(decoder_cell, maximum_iterations):
    """
    Similar to dynamic_rnn but to decode

    Args:
        decoder_cell: (instance of DecoderCell) with step method
        maximum_iterations: (int)
    """
    maximum_iterations = tf.convert_to_tensor(maximum_iterations, dtype=tf.int32)

    # create Tensor Array for outputs by mimicing the structure of decodercell output
    def create_ta(d):
        return tf.TensorArray(
            dtype=d,
            size=0,
            dynamic_size=True)

    initial_time = tf.constant(0, dtype=tf.int32)
    initial_outputs_ta = nest.map_structure(create_ta, decoder_cell.output_dtype)
    initial_state = decoder_cell.initial_state()

    def condition(time, unused_outputs_ta, unused_state):
        return tf.less(time, maximum_iterations)

    def body(time, outputs_ta, state):
        new_output, new_state = decoder_cell.step(time, state)
        outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                      outputs_ta, new_output)

        return (time + 1, outputs_ta, new_state)

    with tf.variable_scope("rnn"):
        res = tf.while_loop(
            condition,
            body,
            loop_vars=[initial_time, initial_outputs_ta, initial_state])

    # get final outputs and states
    final_outputs_ta, final_state = res[1], res[2]

    # unfold and stack the structure from the nested tas
    final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

    # finalize the computation from the decoder cell
    try:
        final_outputs = decoder_cell.finalize(final_outputs)
    except NotImplementedError:
        pass

    # transpose the final output
    final_outputs = nest.map_structure(transpose_batch_time, final_outputs)

    return final_outputs, final_state

