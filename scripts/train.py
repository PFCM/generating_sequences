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


def main(_):
    """Does all the things that we need to do:
        - gets data
        - sets up the graph for inference and sampling
        - gets training ops etc.
        - initialise or reload the variables.
        - train until it's time to go.
    """
    rnn_model = get_forward(inputs, len(vocab))


if __name__ == '__main__':
    tf.app.run()
