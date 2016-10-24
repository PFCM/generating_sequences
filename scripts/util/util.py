"""Utilities that make life easier."""
import tensorflow as tf


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
