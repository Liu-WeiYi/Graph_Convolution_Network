#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Define hyper_parameters
  Created:  09/25/17
"""
from __future__ import division
from __future__ import print_function
import argparse

"""
IMPORTANT Hyper Parameters
"""

def arg_parser():
    parser = argparse.ArgumentParser()

    """ --- Dataset Related --- """
    parser.add_argument('--Dataset', type=str, default='kara_club',
        help="Choose dataset. [kara_club]"
    )
    """ --- Model Check Point --- """
    parser.add_argument('--use_ckpt', type=bool, default=False,
        help="Use pre-trained model"
    )
    parser.add_argument('--ckpt_path', type=str, default='kara_club_log/',
        help="Check point directory to restore"
    )
    """ --- GCN-related Hyper Parameters --- """
    parser.add_argument('--epoch', type=int, default=200,
        help="Number of Epochs to train [200]"
    )
    parser.add_argument('--init_lr', type=float, default=0.01,
        help="initial learning rate [0.1]"
    )
    parser.add_argument('--lr_decay_rate', type=float, default=0.96,
        help="learning rate decay value [0.1]"
    )
    parser.add_argument('--lr_decay_step', type=int, default=10000,
        help="learning rate decay step [5000]"
    )
    parser.add_argument('--dropout', type=float, default=0.5,
        help="Dropout rate [0.5]"
    )
    # TODO: Hidden Layer Output... Is there only ONE Hidden Layer???
    parser.add_argument('--hidden1', type=int, default=16,
        help='Number of units in hidden layer 1.'
    )
    # TODO: What's meaning for Early Stop
    parser.add_argument('--early_stop', type=int, default=10,
        help="Tolerance for early stopping (# of epochs)."
    )

    return parser.parse_args()




