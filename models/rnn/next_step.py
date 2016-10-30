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

from models.rnn.helpers import argmax_and_embed, sample_and_embed


def standard_nextstep_inference(cell, inputs, output_size, scope=None,
                                return_states=False):
    """Gets the forward pass of a standard model for next step prediction.

    Args:
        cell (tf.nn.rnn_cell.RNNCell): the cell to use at each step. For
            multiple layers, wrap them up in tf.nn.rnn_cell.MultiRNNCell.
        inputs: [sequence_length, batch_size, num_features] float input tensor.
            If the inputs are discrete, we expect this to already be embedded.
        output_size (int): the size we project the output to (ie. the number of
            symbols in the vocabulary).
        scope (Optional): variable scope, defaults to "rnn".
        return_states (Optional[bool]): whether or not to return the states
            as well as the projected logits.

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

        states, final_state = tf.nn.rnn(
            cell, inputs, initial_state=initial_state, dtype=tf.float32,
            scope=scope)

        # we want to project them all at once, because it's faster
        logits = tf.concat(0, states)

        with tf.variable_scope('output_layer'):
            # output layer needs weights
            weights = tf.get_variable('weights',
                                      [cell.output_size, output_size])
            biases = tf.get_variable('bias',
                                     [output_size])
            logits = tf.nn.bias_add(tf.matmul(logits, weights), biases)
        # and split them back up to what we expect
        logits = tf.split(0, len(inputs), logits)

        if return_states:
            logits = (states, logits)

    return [initial_state], logits, [final_state]


def standard_nextstep_sample(cell, inputs, output_size, embedding, scope=None,
                             argmax=False, softmax_temperature=1):
    """Generate samples from the standard next step prediction model.
    Assumes that we are modelling sequence of discrete symbols.

    Args:
        cell (tf.nn.rnn_cell.RNNCell): a cell to reproduce the model.
        inputs: input variable, all but the first is ignored.
        output_size (int): the size of the vocabulary.
        embedding: the embedding matrix used.
        scope (Optional): variable scope, needs to line up with what was used
            to make the model for inference the first time around.
        argmax (Optional[bool]): if True, then instead of sampling we simply
            take the argmax of the logits, if False we put a softmax on
            first. Defaults to False.
        softmax_temperature (Optional[bool]): the temparature for softmax.
            The logits are divided by this before feeding into the softmax:
            a high value means higher entropy. Default 1.

    Returns:
        tuple: (initial_state, sequence, final_state) where
            - *initial_state* is a variable for the starting state of the net.
            - *sequence* is a list of len(inputs) length containing the sampled
                symbols.
            - *final_state* the finishing state of the network, to pass along.
    """
    # have to be quite careful to ensure the scopes line up
    with tf.variable_scope(scope or 'rnn') as scope:
        inputs = tf.unpack(inputs)
        batch_size = inputs[0].get_shape()[0].value
        initial_state = cell.zero_state(batch_size, tf.float32)

        # get the output weights
        with tf.variable_scope('output_layer'):
            weights = tf.get_variable('weights',
                                      [cell.output_size, output_size])
            biases = tf.get_variable('bias',
                                     [output_size])

        # choose an appropriate loop function
        sequence = []
        if argmax:
            loop_fn = argmax_and_embed(embedding, output_list=sequence,
                                       output_projection=(weights, biases))
        else:
            loop_fn = sample_and_embed(embedding, softmax_temperature,
                                       output_list=sequence,
                                       output_projection=(weights, biases))

        all_states, fstate = tf.nn.seq2seq.rnn_decoder(
            inputs, initial_state, cell, loop_function=loop_fn, scope=scope)

    return [initial_state], sequence, [fstate]
