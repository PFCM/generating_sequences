"""Helpful functions for both types of rnn model."""
import tensorflow as tf


def _maybe_project(var, proj):
    """Perform a projection of proj is not none."""
    if proj is not None:
        return tf.nn.bias_add(tf.matmul(var, proj[0]), proj[1])
    return var


def argmax_and_embed(embedding, output_projection=None):
    """Returns a callable (usable as a loop_fn for seq2seq) which takes the
    argmax of a batch of outputs and embeds them. Optionally applies a
    projection first.

    Args:
        embedding: an embedding matrix to lookup symbols in.
        output_proj (Optional): tuple (weight, biases) used to project outputs.
            If None (default), no projection is performed.

    Returns:
        embedding from embedding.
    """
    def _argmax_embed(prev, i):
        var = _maybe_project(prev, output_projection)
        next_ = tf.nn.embedding_lookup(embedding, tf.argmax(var, 1))
        return next_

    return _argmax_embed


def sample_and_embed(embedding, temperature, output_projection=None):
    """Returns a callable (usable as a loop_fn for seq2seq) which takes a
    sample from a batch of outputs and embeds them. Optionally applies a
    projection first.

    Args:
        embedding: an embedding matrix to lookup symbols in.
        temperature: temperature to control the pointiness of the softmax.
        output_proj (Optional): tuple (weight, biases) used to project outputs.
            If None (default), no projection is performed.

    Returns:
        embedding from embedding.
    """
    def _sample_embed(prev, i):
        var = _maybe_project(prev, output_projection)
        var /= temperature
        next_ = tf.nn.embedding_lookup(
            embedding, tf.multinomial(var, 1))
        return next_

    return _sample_embed
