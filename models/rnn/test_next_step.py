"""Tests for next_step.py

Mostly just making sure we can get models without shape mismatches etc.
"""
import unittest
import tensorflow as tf

import models.rnn.next_step as ns


class TestStandardNextStep(unittest.TestCase):

    def test_get_model(self):
        """Just make sure we can get a model without errors"""
        # TODO(pfcm) nice helpers for setting up/tearing down a graph & sess
        with tf.Graph().as_default():
            inputs = tf.placeholder(tf.float32, [50, 30, 10])
            cell = tf.nn.rnn_cell.BasicRNNCell(32)

            istate, logits, fstate = ns.standard_nextstep_inference(
                cell, inputs, 5)

            # check shapes are as expected
            self.assertEqual(istate.get_shape().as_list(),
                             [30, 32])
            self.assertEqual(len(logits), 50)
            self.assertEqual(logits[0].get_shape().as_list(),
                             [30, 5])
            self.assertEqual(istate[0].get_shape().as_list(),
                             fstate[0].get_shape().as_list())

    def test_return_states(self):
        """Make sure the return_states flag works (in terms of shape)"""
        with tf.Graph().as_default():
            inputs = tf.placeholder(tf.float32, [50, 30, 10])
            cell = tf.nn.rnn_cell.BasicRNNCell(32)

            istate, logits, fstate = ns.standard_nextstep_inference(
                cell, inputs, 5, return_states=True)

            # check shapes are as expected
            self.assertEqual(logits[0][0].get_shape().as_list(),
                             [30, 32])

    def test_sampler_argmax(self):
        """Make sure we can get an argmax sampling model without exceptions"""
        with tf.Graph().as_default():
            inputs = tf.placeholder(tf.float32, [50, 30, 10])
            cell = tf.nn.rnn_cell.BasicRNNCell(32)
            embedding = tf.get_variable('embedding', [12, 10])

            istate, seq, fstate = ns.standard_nextstep_sample(
                cell, inputs, 5, embedding, argmax=True)

            # check shapes are as expected
            self.assertEqual(seq[0].get_shape().as_list(),
                             [30])

    def test_sampler_softmax(self):
        """Make sure we can get an softmax sampling model without exceptions"""
        with tf.Graph().as_default():
            inputs = tf.placeholder(tf.float32, [50, 30, 10])
            cell = tf.nn.rnn_cell.BasicRNNCell(32)
            embedding = tf.get_variable('embedding', [12, 10])

            istate, seq, fstate = ns.standard_nextstep_sample(
                cell, inputs, 5, embedding, argmax=False)

            # check shapes are as expected
            self.assertEqual(seq[0].get_shape().as_list(),
                             [30])
