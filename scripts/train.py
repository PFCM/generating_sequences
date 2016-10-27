"""Train a model. Plan is to try and make this pretty modular, so there'll
be a lot of work done elsewhere with a consistent API so this guy will be
fairly straightforward, marshal the arguments, manage saving summaries and
checkpoints as well as firing off the training loop. The training loop
itself might have to be somewhere else as some models might need a couple
of phases.
"""

import tensorflow as tf

import util
import models.rnn as rnn


tf.app.flags.DEFINE_string('logdir', None, 'Where to save things')
tf.app.flags.DEFINE_string('cell', 'gru', 'the RNN cell to use')
tf.app.flags.DEFINE_string('model', 'nextstep/standard',
                           'how we are making sequences.')

tf.app.flags.DEFINE_integer('width', 128, 'number of hidden units per cell')
tf.app.flags.DEFINE_integer('layers', 1, 'number of cells stacked up')

tf.app.flags.DEFINE_integer('keep_prob', 1.0, 'dropout keep prob for inputs')

tf.app.flags.DEFINE_integer('batch_size', 50, 'size of mini-batches for SGD')
tf.app.flags.DEFINE_integer('sequence_length', 100, 'max length of bptt')
tf.app.flags.DEFINE_string('dataset', 'warandpeace', 'what data?')
tf.app.flags.DEFINE_integer('embedding_size', 32, 'size of symbol embeddings')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'step size')

FLAGS = tf.app.flags.FLAGS


def get_forward(inputs, vocab_size):
    """Gets the forward parts of the model, based on the flags.
    Returns it all in a dict, to avoid passing around too much.
    """
    cell = util.get_cell(FLAGS.cell, FLAGS.width, FLAGS.layers,
                         FLAGS.keep_prob)
    model = {}
    with tf.variable_scope('rnn') as scope:
        if 'nextstep' in FLAGS.model:
            if 'standard' in FLAGS.model:
                forward = rnn.standard_nextstep_inference(
                    cell, inputs, vocab_size, scope=scope)
                scope.reuse_variables()
                sampling = rnn.standard_nextstep_inference(
                    cell, inputs, vocab_size, scope=scope)
        else:
            raise ValueError('Unknown model: {}'.format(FLAGS.model))

    model['inference'] = {'initial_state': forward[0],
                          'logits': forward[1],
                          'final_state': forward[2]}
    model['sampling'] = {'initial_state': sampling[0],
                         'sequence': sampling[1],
                         'final_state': sampling[2]}
    return model


def get_optimiser():
    """Gets an appropriate optimiser according to FLAGS"""
    return tf.train.AdamOptimizer(FLAGS.learning_rate)


def minimise_xent(rnn_output, targets, global_step=None, var_list=None):
    """Gets training ops to minimise the cross entropy between two sequences,
    one of which is float logits (expecting to be softmaxed) and the other
    integer class labels.

    Args:
        rnn_output: list of tensors, the sequence of unnormalised log-probs.
            Expected to have length `sequence_length` and have elements with
            shape `[batch_size, num_classes]`.
        targets: one big integer tensor of shape
            `[sequence_length, batch_size]`.
        global_step (Optional): global step tensor to update along with the
            training.
        var_list (Optional): list of variables to use for training. Defaults to
            tf.trainable_variables().

    Returns:
        loss_op, train_op: an op representing the average cross entropy
            per symbol and an op to run a step of training.
    """
    # we aren't going to do anything fancy here like use weights (yet) so
    # we may as well just stick them all together into one large batch.
    num_classes = rnn_output[0].get_shape()[1].value
    logits = tf.concat(0, rnn_output)

    loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, tf.reshape(targets, [-1]))
    loss_op = tf.reduce_mean(loss_op)
    # may as well summarise
    tf.scalar_summary('xent', loss_op)

    # now get some stuff to actually train
    optimiser = get_optimiser()

    train_op = optimiser.minimize(
        loss_op, global_step=global_step,
        var_list=var_list or tf.trainable_variables())

    return loss_op, train_op


def _start_msg(msg, width=50):
    print('{:-^{}}'.format(msg, width), end='', flush=True)


def _end_msg(msg, width=50):
    print('\r{:~^{}}'.format(msg, width))


def main(_):
    """Does all the things that we need to do:
        - gets data
        - sets up the graph for inference and sampling
        - gets training ops etc.
        - initialise or reload the variables.
        - train until it's time to go.
    """
    _start_msg('getting data')
    data = util.get_data(FLAGS.batch_size, FLAGS.sequence_length,
                         FLAGS.dataset, FLAGS.embedding_size)
    _end_msg('got data')

    _start_msg('getting forward model')
    rnn_model = get_forward(data['rnn_inputs'],
                            data['target_size'])
    _end_msg('got forward model')

    _start_msg('getting train ops')
    # TODO(pfcm): be more flexible with this
    global_step = tf.Variable(0, name='global_step')
    loss_op, train_op = minimise_xent(rnn_model['inference']['logits'],
                                      data['placeholders']['targets'],
                                      global_step=global_step)
    _end_msg('got train ops')



if __name__ == '__main__':
    tf.app.run()
