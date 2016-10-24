"""Models based on next-step prediction.

Goal is to model the probability of the next symbol given all those that
have come before.

This is pretty flexible, capable to handling arbitrary lengths etc.

These will most likely be trained with truncated BPTT, so the general structure
of the methods is that they take a tf.nn.rnn_cell.RNNCell and a
[sequence_length, batch_size, num_features] input tensor (there will be some
embedding helpers for discrete data) and return a tuple of
initial_state, logits, final_state. If you ask nicely, we could probably
return the states at each timestep as well, otherwise they'll be projected to
the output size.
"""
import tensorflow as tf


def standard_nextstep_inference(cell, inputs, output_size, scope=None):
    """Gets the forward pass of a standard model for next step prediction.

    Args:
        cell (tf.nn.rnn_cell.RNNCell): the cell to use at each step. For
            multiple layers, wrap them up in tf.nn.rnn_cell.MultiRNNCell.
        inputs: [sequence_length, batch_size, num_features] float input tensor.
            If the inputs are discrete, we expect this to already be embedded.
        output_size (int): the size we project the output to (ie. the number of
            symbols in the vocabulary).
        scope (Optional): variable scope, defaults to "rnn".

    Returns:
        tuple: (initial_state, logits, final_state)
            - *initial_state* are float tensors representing the first state of
                the rnn, to pass states across batches.
            - *logits* is a list of [batch_size, vocab_size] float tensors
                representing the output of the network at each timestep.
            - *final_state* are float tensors containing the final state of
                the network, evaluate these to get a state to feed into
                initial_state next batch.
    """
    with tf.variable_scope(scope or "rnn") as scope:
        inputs = tf.unpack(inputs)
        batch_size = inputs[0].get_shape()[0].value
        initial_state = cell.zero_state(batch_size, tf.float32)

        logits, final_state = tf.nn.rnn(
            cell, inputs, initial_state=initial_state, dtype=tf.float32,
            scope=scope)

        # stick all the logits together into one large batch
        logits = tf.concat(0, logits)

        # project them to the right shape
        with tf.variable_scope('output_layer'):
            weights = tf.get_variable('weights',
                                      [cell.output_size, output_size])
            biases = tf.get_variable('bias',
                                     [output_size])
            logits = tf.nn.bias_add(tf.matmul(logits, weights), biases)
        # and split them back up to what we expect
        logits = tf.split(0, len(inputs), logits)

    return initial_state, logits, final_state
