# Graph_Convolution_Network
GCN Related Work. Inspired from https://github.com/tkipf/gcn

## Our Graph Convolution Network can deal with:

1. Monoplex Network (a.k.a Singal Network)
2. Multiplex Network (Each layer has exact same number of nodes)
3. Multilayer Network (Each Layer does not need to have same number of nodes)

## Basic Input Type

1. Input adj shape [3D Tensor]: `shape = [num_layer, num_nodes, num_nodes]`
2. Input Features shape [3D Tensor]: `shape = [num_layer, num_nodes, num_features]`