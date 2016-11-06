"""Train a model. Plan is to try and make this pretty modular, so there'll
be a lot of work done elsewhere with a consistent API so this guy will be
fairly straightforward, marshal the arguments, manage saving summaries and
checkpoints as well as firing off the training loop. The training loop
itself might have to be somewhere else as some models might need a couple
of phases.
"""
import random

import numpy as np
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
tf.app.flags.DEFINE_integer('num_epochs', 100, 'how many times through')

tf.app.flags.DEFINE_integer('sample_length', 500,
                            'size of samples to periodically print')
tf.app.flags.DEFINE_integer('save_every', 300, 'how often to save checkpoints')

FLAGS = tf.app.flags.FLAGS


def get_forward(data_dict):
    """Gets the forward parts of the model, based on the flags.
    Returns it all in a dict, to avoid passing around too much.
    """
    cell = util.get_cell(FLAGS.cell, FLAGS.width, FLAGS.layers,
                         FLAGS.keep_prob)
    inputs = data_dict['rnn_inputs']
    embedding = data_dict['embedding_matrix']
    vocab_size = data_dict['target_size']
    model = {}
    with tf.variable_scope('rnn') as scope:
        if 'nextstep' in FLAGS.model:
            if 'standard' in FLAGS.model:
                forward = rnn.standard_nextstep_inference(
                    cell, inputs, vocab_size, scope=scope)
                scope.reuse_variables()
                sampling = rnn.standard_nextstep_sample(
                    cell, inputs, vocab_size, embedding, scope=scope)
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
    # return tf.train.AdamOptimizer(FLAGS.learning_rate)
    return tf.train.RMSPropOptimizer(FLAGS.learning_rate)


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
    logits = tf.concat(0, rnn_output)

    loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, tf.reshape(targets, [-1]))
    loss_op = tf.reduce_mean(loss_op)
    # may as well summarise
    tf.scalar_summary('xent', loss_op)

    # now get some stuff to actually train
    opt = get_optimiser()

    train_op = opt.minimize(
        loss_op, global_step=global_step,
        var_list=var_list or tf.trainable_variables())

    return loss_op, train_op


def _start_msg(msg, width=50):
    print('{:-^{}}'.format(msg, width), end='', flush=True)


def _end_msg(msg, width=50):
    print('\r{:~^{}}'.format(msg, width))


def _fill_feed(data_dict, in_batch, target_batch=None, state_var=None,
               state_val=None):
    """fills a feed dict"""
    feed = {data_dict['placeholders']['inputs']: in_batch}
    if target_batch is not None:
        feed[data_dict['placeholders']['targets']] = target_batch
    if state_var is not None:
        for var, val in zip(state_var, state_val):
            feed[var] = val
    return feed


def _init_nextstep_state(model, data, seed, sess):
    """Gets a state for the RNN having been run on `seed` inputs"""
    # if seed isn't a multiple of sequence_length, it will be an issue
    state_var = model['sampling']['initial_state']
    state = sess.run(state_var)
    for seq_batch in util.iter_chunks(seed, FLAGS.sequence_length):
        results = sess.run(
            model['sampling']['sequence'] + model['sampling']['final_state'],
            _fill_feed(data, seq_batch, state_var=state_var, state_val=state))
        sequence = results[:FLAGS.sequence_length]
        state = results[FLAGS.sequence_length:]
    return state, [sequence[-1]] * FLAGS.sequence_length


def sample(model, data, sess, seed=None):
    """Draws a sample from the model

    Args:
        model (dict): return value of one of the get_model functions.
        data (dict): return from the get_data function.
        sess (tf.Session): a session in which to run the sampling ops.
        seed (Optional): either a sequence to feed into the first few batches
            or None, in which case just the GO symbol is fed in.

    Returns:
        str: the sample.
    """
    if 'inverse_vocab' not in data:
        data['inverse_vocab'] = {b: a for a, b in data['vocab'].items()}

    if seed is not None:
        # run it a bit to get a starting state
        state, inputs = _init_nextstep_state(model, data, seed, sess)
    else:
        # otherwise start from zero
        state = sess.run(model['sampling']['initial_state'])
        inputs = np.array(
            [[data['go_symbol']] * FLAGS.batch_size] * FLAGS.sequence_length)

    seq = []

    # now just roll through
    while len(seq) < FLAGS.sample_length:
        results = sess.run(
            model['sampling']['sequence'] + model['sampling']['final_state'],
            _fill_feed(data, inputs,
                       state_var=model['sampling']['initial_state'],
                       state_val=state))
        seq.extend(results[:FLAGS.sequence_length-2])
        state = results[FLAGS.sequence_length-1:]
        inputs = np.array(
            [seq[-1]] * FLAGS.sequence_length)

    batch_index = random.randint(0, FLAGS.batch_size)
    samp = ''.join([str(data['inverse_vocab'][symbol[batch_index]])
                    for symbol in seq])

    return samp


def do_training(data, model, loss_op, train_op, saver=None):
    """Trains for a while."""
    _start_msg('getting session, possibly restoring')
    sv = tf.train.Supervisor(logdir=FLAGS.logdir, saver=saver,
                             save_model_secs=FLAGS.save_every)

    with sv.managed_session() as sess:
        _end_msg('ready')

        for epoch in range(FLAGS.num_epochs):
            # get the initial state
            state = sess.run(model['inference']['initial_state'])
            print('~~Epoch {}:'.format(epoch+1))
            epoch_loss, epoch_steps = 0, 0
            for in_batch, targ_batch in data['train_iter']():
                feed_dict = _fill_feed(
                    data, in_batch, targ_batch,
                    state_var=model['inference']['initial_state'],
                    state_val=state)
                results = sess.run(
                    [loss_op, train_op] + model['inference']['final_state'],
                    feed_dict)
                batch_loss = results[0]
                state = results[2:]
                epoch_loss += batch_loss
                epoch_steps += 1
                print('\r~~~~({}): {}'.format(epoch_steps, batch_loss), end='',
                      flush=True)
            print('\r~~~Training loss: {}'.format(epoch_loss/epoch_steps))

            print(sample(model, data, sess))


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
    rnn_model = get_forward(data)
    _end_msg('got forward model')

    _start_msg('getting train ops')
    # TODO(pfcm): be more flexible with this
    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver(tf.all_variables(),
                           max_to_keep=1)
    loss_op, train_op = minimise_xent(
        rnn_model['inference']['logits'], data['placeholders']['targets'],
        global_step=global_step)
    _end_msg('got train ops')
    do_training(data, rnn_model, loss_op, train_op, saver=saver)


if __name__ == '__main__':
    tf.app.run()
