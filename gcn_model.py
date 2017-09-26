# coding: utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Graph Convolution Network,
            Inspired from https://github.com/tkipf/gcn
  Created:  09/26/17
"""
import tensorflow as tf
from datetime import datetime

class Graph_Convolution_Network:
    """
    GCN
    """
    def __init__(self,args):
        # Dataset
        self.dataset        = args.dataset
        # model operator
        self.use_ckpt       = args.use_ckpt
        self.ckpt_path      = args.ckpt_path
        self.train          = args.train
        self.test           = args.test
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
        self.adj_ph     = tf.placeholder(dtype=tf.float32, shape=[None,None])
        self.feature_ph = tf.placeholder(dtype=tf.float32, shape=[None,None])

