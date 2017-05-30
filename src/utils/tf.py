import tensorflow as tf


def conv2d(inputs, filters=64, kernel_size=3, strides=1, padding='SAME'):
    return tf.layers.conv2d(inputs, filters, kernel_size, strides, padding, activation=tf.nn.relu)


def max_pooling2d(inputs, pool_size=2, strides=2, padding="SAME"):
    return tf.layers.max_pooling2d(inputs, pool_size, strides, padding)


def batch_normalization(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training, axis=-1) 


def dynamic_decode(decoder_cell, maximum_iterations):
    """
    Similar to dynamic_rnn but to decode

    Args:
        decoder_cell: (instance of DecoderCell) with step method
        maximum_iterations: (int)
    """
    maximum_iterations = tf.convert_to_tensor(maximum_iterations, dtype=tf.int32)

    initial_time = tf.constant(0, dtype=tf.int32)
    initial_outputs_ta = tf.TensorArray(dtype=tf.float32, size=maximum_iterations)
    initial_state = decoder_cell.initial_state()

    def condition(time, unused_outputs_ta, unused_state):
        return tf.less(time, maximum_iterations)

    def body(time, outputs_ta, state):
        new_y, new_state = decoder_cell.step(time, state)
        outputs_ta = outputs_ta.write(time, new_y)
        return (time + 1, outputs_ta, new_state)

    with tf.variable_scope("rnn"):
        res = tf.while_loop(
            condition,
            body,
            loop_vars=[initial_time, initial_outputs_ta, initial_state])

    final_outputs_ta = res[1].stack()
    final_outputs_ta = tf.transpose(final_outputs_ta, (1, 0, 2))

    final_state = res[2]

    return final_outputs_ta, final_state

