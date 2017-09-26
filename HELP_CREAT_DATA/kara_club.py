#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Help Creating Kara_Club Data/Labels for Training and Testing
  Created:  09/25/17
"""
from __future__ import division
from __future__ import print_function

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

def create_kara_club_data():
    """
    Create kara club data to disk

    @return adj, y_label
    """
    G = nx.karate_club_graph()
    adj = nx.to_numpy_matrix(G)

    Community_INFO = [
        [0,1,2,3,7,9,11,12,13,17,19,21],
        [4,5,6,10,16],
        [24,25,28,31],
        [8,14,15,18,20,22,23,26,27,29,30,32,33]
    ]
    # assign Labels ~~
    labels_len = len(Community_INFO)
    node_label = np.zeros(shape=(len(G.nodes()),labels_len),dtype=np.float)
    idx = 0
    for com in Community_INFO:
        seed = random.sample(com,1)[0]
        seed_idx = G.nodes().index(seed)
        seed_feature = np.zeros(shape=(labels_len))
        seed_feature[idx] = 1
        node_label[seed_idx,:] = seed_feature
        idx += 1

    # write to disk
    pickle.dump(adj, open('../data/kara_club/adj.pkl','wb'))
    pickle.dump(node_label, open('../data/kara_club/features.pkl','wb'))

    return adj, node_label


if __name__ == "__main__":
    create_kara_club_data()
    print('down')
