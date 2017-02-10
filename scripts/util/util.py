"""Utilities that make life easier."""
from functools import partial

import numpy as np
import tensorflow as tf

import util.text_file as text_file


def _batch_iterator(data_items, batch_size, shuffle=True, transpose=True):
    """iterates data along its first dimension"""
    num_items = data_items[0].shape[0]
    for item in data_items[1:]:
        if item.shape[0] != num_items:
            raise ValueError(
                'all data needs to be the same shape in the first dimension')
    num_batches = num_items // batch_size
    indices = np.arange(num_items)
    if shuffle:
        np.random.shuffle(indices)

    for i in range(num_batches):
        # grab a set of indices
        batch_indices = indices[i*batch_size:(i+1)*batch_size]
        if transpose:
            yield tuple((item[batch_indices, ...].T for item in data_items))
        else:
            yield tuple((item[batch_indices, ...] for item in data_items))


def _integer_placeholders(batch_size, sequence_length):
    """Get placeholders for when input/target are integers"""
    inps = tf.placeholder(tf.int32, shape=[sequence_length, batch_size],
                          name='inputs')
    targ = tf.placeholder(tf.int32, shape=[sequence_length, batch_size],
                          name='targets')

    return inps, targ


def _make_targets(data):
    """for the datasets which just return a single item"""
    return data[:-1], data[1:]


def _get_result(num, func, *args):
    """Returns a function which calls another function and returns a particular
    element of its results."""
    def _get():
        return func(*args)[num]

    return _get


def _get_result_make_targets_and_yield(num, func, *args):
    """Gets an appropriate iterator"""
    for batch in func(*args)[num]:
        yield _make_targets(batch)


def embed(int_inputs, embedding_size, num_inputs, scope=None):
    """Embeds an integer tensor.

    Args:
        int_inputs: the tensor of indices.
        embedding_size (int): the size of the embeddings required.
        num_inputs (int): the maximal number of input ids.
        scope (Optional): a scope under which to add the ops/get the embedding
            matrix.

    Returns:
        embedded tensor
    """
    with tf.variable_scope(scope or 'embedding'):
        embeddings = tf.get_variable('embeddings',
                                     shape=[num_inputs, embedding_size])
        return tf.nn.embedding_lookup(embeddings, int_inputs), embeddings


def get_data(batch_size, sequence_length, dataset, embedding_size):
    """Gets a dict with the things needed for the data, including placeholders

    Args:
        batch_size (int): sequences per batch.
        sequence_length (int): length of sequences concerned (for data where it
            matters).
        dataset (str): the dataset. Options at this stage are:
            - _warandpeace_: character-level war and peace.
            - _occult_: a set of occult texts.
            - _ptb/char_: character level penn treebank
            - _ptb/word_: word level penn treeback
            - _mnist_: sequential mnist (not sigmoid)
            - _jsb_: JSB chorales
            - _txt:path/to/textfile_: arbitrary text file with sequences
                separated by lines.
        embedding_size: if the data needs embedding, how big should they be?

    Returns:
        dict: fields:
            - `placeholders`: dict with placeholder variables:
                - `inputs`: for the inputs
                - `targets`: for targets, probably the inputs shifted across.
            - `rnn_inputs`: the tensor to use as inputs for the RNN, probably
                placeholder/inputs through an embedding lookup.
            - `train_iter`: callable that returns an iterator over training
                data
            - `valid_iter`: callable that returns an iterator over valid data
            - `test_iter`: callable that returns an iterator over test data.
            - `vocab`: iff the data is discrete symbols, a dict mapping ids to
                some more sensible representation.
            - `target_size`: probably len(vocab), but maybe not.
    """
    data_dict = {}
    lengths = None
    if dataset == 'warandpeace':
        import rnndatasets.warandpeace as data
        vocab = data.get_vocab('char')
        inputs, targets = _integer_placeholders(batch_size, sequence_length)
        rnn_inputs, embedding = embed(
            inputs, embedding_size, len(vocab), scope='input')

        train_fetcher = partial(_get_result_make_targets_and_yield, 0,
                                data.get_split_iters, sequence_length + 1,
                                batch_size)
        valid_fetcher = partial(_get_result_make_targets_and_yield, 1,
                                data.get_split_iters, sequence_length + 1,
                                batch_size)
        test_fetcher = partial(_get_result_make_targets_and_yield, 2,
                               data.get_split_iters, sequence_length + 1,
                               batch_size)
        data_dict['target_size'] = len(vocab)
        data_dict['embedding_matrix'] = embedding
        go_symbol = vocab['<GO>']
    elif dataset == 'occult':
        # a lot of this is (will be) shared, should refactor into functions
        import rnndatasets.occult as data
        train, valid, test, vocab = data.occult_raw_data()
        inputs, targets = _integer_placeholders(batch_size, sequence_length)
        rnn_inputs, embedding = embed(
            inputs, embedding_size, len(vocab), scope='input')
        # "partial" even though we're providing all the args
        train_fetcher = partial(data.batch_iterator, train, batch_size,
                                sequence_length, time_major=True)
        valid_fetcher = partial(data.batch_iterator, valid, batch_size,
                                sequence_length, time_major=True)
        test_fetcher = partial(data.batch_iterator, test, batch_size,
                               sequence_length, time_major=True)
        data_dict['target_size'] = len(vocab)
        data_dict['embedding_matrix'] = embedding
        go_symbol = vocab['<GO>']
    elif dataset == 'ptb/word':
        import rnndatasets.ptb as data
        train, valid, test, vocab = data.get_ptb_data()
        inputs, targets = _integer_placeholders(batch_size, sequence_length)
        rnn_inputs, embedding = embed(
            inputs, embedding_size, len(vocab), scope='input')
        # "partial" even though we're providing all the args
        train_fetcher = partial(data.batch_iterator, train, batch_size,
                                sequence_length, time_major=True)
        valid_fetcher = partial(data.batch_iterator, valid, batch_size,
                                sequence_length, time_major=True)
        test_fetcher = partial(data.batch_iterator, test, batch_size,
                               sequence_length, time_major=True)
        data_dict['target_size'] = len(vocab)
        data_dict['embedding_matrix'] = embedding
        go_symbol = vocab['<GO>']
    elif dataset == 'mnist':
        raise NotImplementedError('not even sure this one is a good idea')
    elif dataset == 'jsb':
        raise NotImplementedError('nerp')
    elif 'txt' in dataset:
        # arbitrary line separated text file
        fpath = dataset.split(':')[-1]
        data, lengths_data, vocab = text_file.get_text_file_data(
            fpath, sequence_length)
        inputs, targets = _integer_placeholders(batch_size, sequence_length)
        lengths = tf.placeholder(tf.int32, shape=[batch_size], name='lengths')
        rnn_inputs, embedding = embed(
            inputs, embedding_size, len(vocab))
        # TODO valid, test
        valid_fetcher = None
        test_fetcher = None
        input_data = data[:, :-1]
        target_data = data[:, 1:]
        # get a callable which yields training batches
        train_fetcher = partial(_batch_iterator,
                                (input_data, target_data, lengths_data),
                                batch_size, shuffle=True)
        go_symbol = vocab['<GO>']
        data_dict['target_size'] = len(vocab)
        data_dict['embedding_matrix'] = embedding
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

    data_dict.update({
        'placeholders': {
            'inputs': inputs,
            'targets': targets,
            'lengths': lengths},
        'rnn_inputs': rnn_inputs,
        'train_iter': train_fetcher,
        'valid_iter': valid_fetcher,
        'test_fetcher': test_fetcher,
        'go_symbol': go_symbol})
    if vocab:
        data_dict['vocab'] = vocab

    return data_dict


def get_cell(name, width, layers=1, keep_prob=1.0):
    """Gets a tf.nn.rnn_cell.RNNCell to make models. Note that we always let
    the state be a tuple, so you have to be a little bit careful.

    Args:
        cell (str): the name of the cell. At this stage, expected options are:
            - `vanilla` for a classic tanh cell.
            - `lstm` for a fairly standard lstm.
            - `gru` for a gated recurrent unit.
        width (int): how many hidden units per cell.
        layers (Optional[int]): how many of the cells there are, stacked on top
            of one another (thus far in a standard manner). Default is 1.
        keep_prob (Optional): float or tensor with the dropout applied to
            the _input_ of each layer. If 1.0 no dropout is applied.

    Returns:
        tf.nn.rnn_cell.RNNCell: the cell (or a MultiRNNCell if multiple
            layers.)

    Raises:
        ValueError: if the cell is not listed above.
    """
    if name == 'vanilla':
        cell = tf.nn.rnn_cell.BasicRNNCell(width)
    elif name == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell(width, state_is_tuple=True)
    elif name == 'gru':
        cell = tf.nn.rnn_cell.GRUCell(width)
    else:
        raise ValueError('Cell `{}` unknown'.format(name))

    if keep_prob != 1.0:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)

    if layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers,
                                           state_is_tuple=True)

    return cell
