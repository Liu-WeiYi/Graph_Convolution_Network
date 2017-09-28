#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Define a bunch of utils for GCN
  Created:  09/27/17
"""
import traceback
import tensorflow as tf

def sparse_dropout(x, keep_prob, num_of_non_zero_features):
    """
    Dropout for sparse tensors. Inspired from gcn

    @param keep_prob: keep probality
    @param num_of_non_zero_features: none zero features number
    """
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(num_of_non_zero_features)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)



def gcn_conv(name, reuse, adjs_input, features_input, keep_prob,  num_of_non_zero_features, num_labels):
    """
    conv graph

    @param name: current conv layer name
    @param reuse: reuse_variable Flag
    @param adjs_input: network (a 3D tensor)
    @param features_input: features (a 3D tensor)
    @param keep_prob: keep prob for dropout layer
    @param num_of_non_zero_features: non_zero features
    @param num_labels: number of output labels

    return conved_layer
    """
    with tf.variable_scope(name, reuse=reuseModel) as scope:
        if reuse is True:
            scope.reuse_variables()

        num_layers, num_nodes, _ = adjs_input.get_shape()

        for layer in range(num_layers):
            adj     = adjs_input[layer, ...]
            feature = features_input[layer, ...]

            weight = tf.get_variable(
                name='w1_%d'%layer,
                shape=[num_nodes, num_labels],
                initializer=tf.truncated_normal_initializer()
            )
            bias = tf.get_variable(
                name='b1_%d'%layer,
                shape=[num_labels],
                initializer=tf.constant_initializer()
            )
            """ TODO:  LOG Weights and Bias!! """
            # 1. Drop Out First ---  https://github.com/tkipf/gcn
            drop_out_input = sparse_dropout(
                x=feature,
                keep_prob=keep_prob,
                num_of_non_zero_features=num_of_non_zero_features
            )

            # 2. Multiple Adj with its Features, and add bias
            for layer in range(num_layers):
                # Feature * Weight
                F_W = tf.matmul(features_input[layer,...],weight)
                # Graph * F_W
                conved_graph = tf.matmul(adj, F_W)
                # add bias
                conved_graph = tf.nn.bias_add(conved_graph,bias)

            # update adjs_input
            adjs_input[layer] = conved_graph

        # add activation function
        return tf.nn.relu(adjs_input)
