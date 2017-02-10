"""some utilities for handling text files"""
import os
import collections

import numpy as np


def split_fold(data, max_size):
    """checks elements of a list are at most a particular size, if greater,
    splits them."""
    new_data = []
    for line in data:
        if len(line) > max_size:
            line = [line[0+i:max_size+i]
                    for i in range(0, len(line), max_size)]
        else:
            line = [line]
        new_data.extend(line)
    return new_data


def load_vocab(vocab_file):
    """loads in a vocab tsv.

    Args:
        vocab_file (str): the file to try and load

    Returns:
        dict: the vocab, mapping str -> int.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    print('looking for vocab file: {}'.format(vocab_file))
    with open(vocab_file, 'r') as fp:
        all_vocab = fp.read()
    print('found it')
    # split on newlines
    all_vocab = all_vocab.split('\n')
    # and split on tabs
    all_vocab = [item.split('\t') for item in all_vocab]
    # filter -- if any have length 3, the character must have been a tab
    all_vocab = [[item[0], '\t'] if len(item) == 3 else item
                 for item in all_vocab]
    # filter -- if any are [''] then the preceding item must have been a '\n'
    filtered_vocab = []
    for i, item in enumerate(all_vocab):
        if item == [''] and i > 0:
            filtered_vocab[-1][1] = '\n'
        else:
            filtered_vocab.append(item)
    # now make it into a dict, and turn the values into integers
    return {b: int(a) for a, b in filtered_vocab}


def create_vocab(data):
    """makes a vocab from a string. Assigns ids by counting.

    Args:
        data (str): the data to count. This could really be any iterable.

    Returns:
        dict: element of `data` -> int
    """
    counter = collections.Counter(data)
    return {a: i for i, (a, b) in enumerate(counter.most_common())}


def load_file_and_vocab(filepath):
    """Loads a file and make a vocab dictionary.

    Args:
        filepath (str): path to a file containing unicode text.

    Returns:
        str: the contents of the file
        dict: a dictionary of str -> int mapping ids to characters in the text.
            IDs are assigned in one of two ways: if there is a file called
            `vocab.tsv` in the same directory as the text file then we will
            read it, assuming it has two columns and no header, the first of
            which are the ids, the second the characters. If such a file does
            not exist then IDs are assigned based on the frequency of the
            characters in the file. These assignments are not written to disk,
            so don't change the file in between if you need them to be
            consistent.
    """
    with open(filepath, 'r') as fp:
        data_str = fp.read()
    # now check for vocab
    try:
        vocab = load_vocab(os.path.join(os.path.dirname(filepath),
                                        'vocab.tsv'))
    except FileNotFoundError:
        print('no vocab found, generating')
        vocab = create_vocab(data_str)
    return data_str, vocab


def _newline_split(data):
    return data.split('\n')


def make_data_sequences(data_str, vocab, sequence_splitter=_newline_split,
                        token_splitter=list):
    """convert the data a from a big string into a list of lists of integers.

    Args:
        data_str (str): data, as one long string.
        vocab (dict): mapping from str->int which we use to convert sequences
            into ids.
        sequence_splitter (callable): a function which when called on the data
            string returns an iterable which will yield one sequence at a time.
        token_splitter (callable): a function which when called on a single
            sequence will split it into a list of tokens which we can
            substitute for ids.

    Returns:
        list: the data as a list of lists of ints.
    """
    data = [token_splitter(item) for item in sequence_splitter(data_str)]
    # now convert to ids
    data = [[vocab[token] for token in item] for item in data]
    return data


def _pad(seq, max_size, go_id, pad_id):
    """pads to be at most max_size and then prepends the go symbol.
    Results in sequences of max_size +1"""
    if len(seq) < max_size:
        padding = [pad_id] * (max_size-len(seq))
        seq += padding
    seq.insert(0, go_id)
    return seq


# TODO shuffle it and grab some validation sequences
def get_text_file_data(filepath, max_size):
    """Gets a text file as ndarray of ints, treating each line as a distinct
    item (although chopping them to a maximum length).

    Items smaller than `max_size` will be padded with an appropriate symbol.

    Args:
        filepath (str): where to find the data
        max_size (int): the maximum length of each line.

    Returns:
        data (ndarray): ndarray of int32 containing the data in batch-major
            order (shape [num_items, max_size+1] because all sequences have the
            '<GO>' symbol prepended).
        lengths (ndarray): array containing the actual length of each example
            before padding, shape [num_items]
        vocab (dict): the mapping from int->token in the file. Currently
            all tokens are just single characters (apart from `<GO>` and
            `<PAD>`).
    """
    # first step is get the data and vocab
    data_str, vocab = load_file_and_vocab(filepath)
    # use the vocab to split up the data and turn it into ints
    data = make_data_sequences(data_str, vocab)
    # trim excessively long sequences
    data = split_fold(data, max_size)
    # gather the current lengths of the sequences
    lengths = [len(seq) for seq in data]
    # make sure the vocab has GO and PAD tokens
    if '<GO>' not in vocab:
        vocab['<GO>'] = len(vocab)
    if '<PAD>'not in vocab:
        vocab['<PAD>'] = len(vocab)
    # now pad the sequences if required
    data = [_pad(seq, max_size, vocab['<GO>'], vocab['<PAD>']) for seq in data]
    # batch up into ndarrays as promised
    data = np.array(data, dtype=np.int32)
    lengths = np.array(lengths, dtype=np.int32)

    return data, lengths, vocab
