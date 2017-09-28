#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Define a bunch of utils for GCN
  Created:  09/27/17
"""
import traceback
import tensorflow as tf

def gcn_conv(name, reuse, inputs):
    """
    conv graph

    @param name: current conv layer name
    @param reuse: reuse_variable Flag
    @param inputs: network (a 3D tensor)

    return conved_layer
    """
    with tf.variable_scope(name, reuse=reuseModel) as scope:
        if reuse is True:
            scope.reuse_variables()

        weight = tf.get_variable(
            name='w1',
            shape=[inputs.get_shape()[0], self.hidden1],
            initializer=tf.truncated_normal_initializer()
        )
        bias = tf.get_variable(
            name='b1',
            shape=[self.hidden1],
            initializer=tf.constant_initializer()
        )
        """ TODO:  LOG Weight and Bias!! """

        # 1. Drop Out First ---  https://github.com/tkipf/gcn

        # 2. Multiple Adj with its Features



