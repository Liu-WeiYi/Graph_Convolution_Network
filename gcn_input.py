#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Preparing Data
  Created:  09/25/17
"""

import sys
import os
import pickle
import traceback

import numpy as np

def _process_adj(A):
    """
    For each layer, we process adj by using D*A*D

    @param A: adj
    @return A_: D^{-0.5}adjD^{+0.5}
    """
    num_layers, num_nodes, _ = A.shape

    A_ = np.zeros(shape=A.shape)

    for layer in range(num_layers):
        layer_A = A[layer, ...]
        # 1. add Identity Matrix
        layer_A = layer_A + np.identity(num_nodes, np.float)
        # 2. symmetric normalization A with Diagonal Node Degree Matrix D
        D_flatten = np.sum(layer_A,axis=-1)
        D_left = np.diag(np.power(D_flatten,-0.5)) # D^{-0.5}
        D_right = np.diag(np.power(D_flatten,0.5)) # D^{0.5}
        # 3. A_ = D^{-0.5}*A*D^{0.5}
        layer_A_ = np.matmul(np.matmul(D_left,layer_A), D_right)
        # 4. stacked into A_
        A_[layer] = layer_A_

    return A_


def kara_club(path, net_version):
    """
    preparing adj adj and features for kara club

    @param path: dataset (self.dataset)
    @param net_version: define network version
    @return processed_adj D^{-0.5}adjD^{+0.5}
            features
    """
    try:
        assert net_version == "Monoplex"
    except Exception as exc:
        print(exc)
        sys.exit('*** FATAL ERROR: NET VERSION SHOULD BE [Monoplex].\n*** HOWEVER, IT IS [%s]\n'%net_version)

    try:
        adj_file = os.path.join('data',path,'adj.pkl')
        feature_file = os.path.join('data',path,'features.pkl')
    except Exception as exc:
        print(traceback.format_exc())
        print(exc)
        sys.exit('*** FATAL ERROR: CANNOT OPEN DATA')

    A = pickle.load(open(adj_file,'rb'))
    features = pickle.load(open(feature_file,'rb'))

    # In case mix np.matrix and np.array, we transfer all to narray
    A = np.array(A)
    features = np.array(features)
    # as for monoplex network, we need to expand dim
    A = np.expand_dims(A,0)
    features = np.expand_dims(features,0)

    return _process_adj(A), features

if __name__ == "__main__":
    A, features = kara_club("kara_club","Monoplex")
    print(A)
    print(features)
