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
        self.adj_ph     = tf.placeholder(dtype=tf.float32, shape=[None,None,None])
        self.feature_ph = tf.placeholder(dtype=tf.float32, shape=[None,None,None])

        self.__model_construction(inputs=self.adj_ph, reuseModel=False)

    def __model_construction(self, inputs, reuseModel=False):
        """
        Construct Model

        @param inputs: 3D Tensor, shape=[num_layers, num_nodes, num_nodes]
        @param reuseModel: [False]

        @return logits: constructed logits
        """
        layer = []

        # First GCN Layer
        layer_conved = gcn_conv(
            name='GCN_1',
            reuse=reuseModel
        )




