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
    @param A: adj
    @return A_: D^{-0.5}adjD^{+0.5}
    """
    shape = A.shape[0]
    # 1. add Identity Matrix
    A = A + np.identity(shape, np.float)
    # 2. symmetric normalization A with Diagonal Node Degree Matrix D
    D_flatten = np.sum(A,axis=-1)
    D_left = np.diag(np.power(D_flatten,-0.5)) # D^{-0.5}
    D_right = np.diag(np.power(D_flatten,0.5)) # D^{0.5}
    # 3. A_ = D^{-0.5}*A*D^{0.5}
    A_ = np.matmul(np.matmul(D_left,A), D_right)

    return A_


def kara_club(path):
    """
    preparing adj adj and features for kara club

    @param path: dataset (self.dataset)
    @return processed_adj D^{-0.5}adjD^{+0.5}
            features
    """
    try:
        adj_file = os.path.join('data',path,'adj.pkl')
        feature_file = os.path.join('data',path,'features.pkl')
    except Exception as exc:
        print(traceback.format_exc())
        print(exc)
        sys.exit('FATAL ERROR: CANNOT OPEN DATA')

    A = pickle.load(open(adj_file,'rb'))
    feature = pickle.load(open(feature_file,'rb'))

    # In case mix np.matrix and np.array, we transfer all to narray
    A = np.array(A)
    feature = np.array(feature)

    return _process_adj(A), feature

if __name__ == "__main__":
    A, features = kara_club("kara_club")
    print(A)
    print(features)
