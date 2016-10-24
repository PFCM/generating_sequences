"""Tests for next_step.py

Mostly just making sure we can get models without shape mismatches etc.
"""
import unittest
import tensorflow as tf

import models.rnn.next_step as ns


class TestStandardNextStep(unittest.TestCase):

    def test_get_model(self):
        """Just make sure we can get a model without errors"""
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
        self.assertEqual(istate.get_shape().as_list(),
                         fstate.get_shape().as_list())
