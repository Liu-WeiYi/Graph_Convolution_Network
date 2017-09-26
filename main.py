#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Program Access
  Created:  09/26/17
"""
from __future__ import division
from __future__ import print_function

from Hyper_Parameters import arg_parser
from gcn_model import Graph_Convolution_Network

def main(args):
    """
    call GCN with hyper parameters

    @param parser: hyper_paramters class
    """
    print('===========\nInit GCN...\n===========')
    gcn = Graph_Convolution_Network(args)

    if args.train is True:
        print('===========\nTrain GCN...\n===========')

    if args.test is False:
        print('===========\nTesting GCN...\n===========')


if __name__ == "__main__":
    args = arg_parser()
    main(args)
    print('All Finished :)')