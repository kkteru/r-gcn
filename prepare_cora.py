import numpy as np
import scipy.sparse as sp
import torch
import os
import sys
import pickle as pkl
import pdb

path = "./data/cora/"
dataset = "cora"


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


print('Loading {} dataset...'.format(dataset))

idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                    dtype=np.dtype(str))
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
labels = encode_onehot(idx_features_labels[:, -1])
pdb.set_trace()

# build graph
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                dtype=np.int32)
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                 dtype=np.int32).reshape(edges_unordered.shape)
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(labels.shape[0], labels.shape[0]),
                    dtype=np.float32)

# build symmetric adjacency matrix
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

features = normalize(features)
adj = normalize(adj + sp.eye(adj.shape[0]))

train_idx = range(140)
valid_idx = range(200, 500)
test_idx = range(500, 1500)

features = torch.FloatTensor(np.array(features.todense()))
# labels = torch.LongTensor(np.where(labels)[1])

train_idx = torch.LongTensor(train_idx)
valid_idx = torch.LongTensor(valid_idx)
test_idx = torch.LongTensor(test_idx)

data = {'A': [adj],
        'feat': features,
        'y': labels,
        'train_idx': train_idx,
        'valid_idx': valid_idx,
        'test_idx': test_idx
        }

dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

with open(dirname + '/' + dataset + '.pickle', 'wb') as f:
    pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
