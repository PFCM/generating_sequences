"""tests for functions in helpers.py"""
from nose2.tools.decorators import with_setup, with_teardown

import numpy as np
import tensorflow as tf

import models.rnn.helpers as helpers


def setup_session():
    """Clears the default graph and starts an interactive session"""
    tf.reset_default_graph()
    tf.InteractiveSession()


def teardown_session():
    """Closes the session"""
    tf.get_default_session().close()


@with_setup(setup_session)
@with_teardown(teardown_session)
def test_argmax_and_embed():
    """Ensure argmax_and_embed works without projection"""
    embedding = tf.get_variable('embedding', [3, 20])
    data = tf.get_variable('input', initializer=np.array([[1., 2., 1.]]))

    loop_fn = helpers.argmax_and_embed(embedding, output_projection=None)
    correct = tf.nn.embedding_lookup(embedding, [1])

    result = loop_fn(data, 0)

    # get ready to see if it's right
    sess = tf.get_default_session()
    sess.run(tf.initialize_all_variables())

    a, b = sess.run([result, correct])

    assert np.all(a == b)


@with_setup(setup_session)
@with_teardown(teardown_session)
def test_sample_and_embed():
    """Ensure sample_and_embed works without projection"""
    embedding = tf.get_variable('embedding', [3, 20])
    data = tf.get_variable('input', initializer=np.array([[1., 2., 1.]]))

    loop_fn = helpers.sample_and_embed(embedding, 1., output_projection=None)
    result = loop_fn(data, 0)

    # get ready to see if does indeed pick out one item
    sess = tf.get_default_session()
    sess.run(tf.initialize_all_variables())

    a, embed_mat = sess.run([result, embedding])

    found = False
    for row in embed_mat:
        if np.all(row == a):
            found = True

    assert found


@with_setup(setup_session)
@with_teardown(teardown_session)
def test_argmax_and_embed_with_projection():
    """Ensure argmax_and_embed works with projection"""
    embedding = tf.get_variable('embedding', [10, 11])
    proj = (tf.get_variable('weights', [3, 10]),
            tf.get_variable('biases', [10]))
    data = tf.get_variable('input', initializer=np.array([[1., 2., 1.]],
                                                         dtype=np.float32))
    loop_fn = helpers.argmax_and_embed(embedding, output_projection=proj)

    # we don't know what the correct answer is now because it's randomly
    # projected, so let's get what we need to do it by hand
    correct_projection = tf.nn.bias_add(tf.matmul(data, proj[0]), proj[1])

    result = loop_fn(data, 0)

    # get ready to see if it's right
    sess = tf.get_default_session()
    sess.run(tf.initialize_all_variables())

    a, embedding, projection = sess.run(
        [result, embedding, correct_projection])

    argmax_p = np.argmax(projection)

    assert np.all(embedding[argmax_p] == a)


@with_setup(setup_session)
@with_teardown(teardown_session)
def test_sample_and_embed_with_projection():
    """Ensure sample_and_embed works with projection"""
    embedding = tf.get_variable('embedding', [10, 11])
    proj = (tf.get_variable('weights', [3, 10]),
            tf.get_variable('biases', [10]))
    data = tf.get_variable('input', initializer=np.array([[1., 2., 1.]],
                                                         dtype=np.float32))

    loop_fn = helpers.sample_and_embed(embedding, 1., output_projection=proj)
    result = loop_fn(data, 0)

    # get ready to see if does indeed pick out one item
    sess = tf.get_default_session()
    sess.run(tf.initialize_all_variables())

    a, embed_mat = sess.run([result, embedding])

    found = False
    for row in embed_mat:
        if np.all(row == a):
            found = True

    assert found
