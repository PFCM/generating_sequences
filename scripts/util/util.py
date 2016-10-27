"""Utilities that make life easier."""
import tensorflow as tf


def get_data(batch_size, sequence_length, dataset):
    """Gets a dict with the things needed for the data, including placeholders

    Args:
        batch_size (int): sequences per batch.
        sequence_length (int): length of sequences concerned (for data where it
            matters).
        dataset (str): the dataset. Options at this stage are:
            - _warandpeace_: character-level war and peace.
            - _ptb/char_: character level penn treebank
            - _ptb/word_: word level penn treeback
            - _mnist_: sequential mnist (not sigmoid)
            - _jsb_: JSB chorales

    Returns:
        dict: fields:
            - `placeholders`: dict with placeholder variables:
                - `input`: for the inputs
                - `targets`: for targets, probably the inputs shifted across.
            - `train`: training data
            - `valid`: validation data
            - `test`: testing data
    """
    if dataset == 'warandpeace':
        raise NotImplementedError('nope')
    elif dataset == 'ptb/char':
        raise NotImplementedError('not yet')
    elif dataset == 'ptb/word':
        raise NotImplementedError('hold your horses')
    elif dataset == 'mnist':
        raise NotImplementedError('not even sure this one is a good idea)
    elif dataset == 'jsb':
        raise NotImplementedError('nerp')
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


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
