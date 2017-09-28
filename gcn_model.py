# coding: utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Graph Convolution Network,
            Inspired from https://github.com/tkipf/gcn
  Created:  09/26/17
"""
import tensorflow as tf
from datetime import datetime

from utils import *

class Graph_Convolution_Network:
    """
    GCN Related
    """
    def __init__(self,args):
        # Dataset
        self.dataset        = args.dataset
        # model operator
        self.use_ckpt       = args.use_ckpt
        self.ckpt_path      = args.ckpt_path
        self.train          = args.train
        self.test           = args.test
        # Multi-Net hyper parameters
        self.net_version    = args.net_version
        self.layer_merging  = args.layer_merging
        # GCN hyper parameters
        self.epochs         = args.epochs
        self.init_lr        = args.init_lr
        self.lr_decay_rate  = args.lr_decay_rate
        self.lr_decay_step  = args.lr_decay_step
        self.dropout        = args.dropout
        self.hidden1        = args.hidden1
        self.early_stop     = args.early_stop

        print('GCN Hyper Parameters:')
        for k in self.__dict__:
            print('--',k, ': ', self.__dict__[k])

        # define Place Holders :)
        self.adj_ph     = tf.sparse_placeholder(dtype=tf.float32)
        self.feature_ph = tf.placeholder(dtype=tf.float32, shape=[None,None,None])
        self.num_of_non_zero_features = tf.placeholder(dtype=tf.int32)

        self.__model_construction(
            adjs_input    = self.adj_ph,
            features_input = self.feature_ph,
            reuseModel    = False
        )

    def __model_construction(self, adjs_input, features_input ,reuseModel=False):
        """
        Construct Model

        @param adjs_input: 3D Tensor, shape=[num_layers, num_nodes, num_nodes]
        @param features_input: 3D Tensor, shape=[num_layers, num_nodes, num_features]
        @param reuseModel: [False]

        @return logits: constructed logits
        """
        layer = [adjs_input]

        # First GCN Layer
        layer_conved = gcn_conv(
            name='GCN_1',
            reuse=reuseModel,
            adjs_input=layer[-1],
            features_input=features_input,
            keep_prob=self.keep_prob,
            num_of_non_zero_features=self.num_of_non_zero_features,
            num_labels=self.hidden1
        )
        layer.append(layer_conved)
        # Second GCN Layer
        layer_conved = gcn_conv(
            name='GCN_2',
            reuse=reuseModel,
            adjs_input=adjs_input,
            features_input=layer[-1],
            keep_prob=self.keep_prob,
            num_of_non_zero_features=self.num_of_non_zero_features,
            num_labels=self.num_labels
        )
