import logging
import random
import pdb

import numpy as np
import scipy.sparse as sp
import torch


def get_torch_sparse_matrix(A, dev):
    '''
    A : list of sparse adjacency matrices
    '''
    idx = torch.LongTensor([A.tocoo().row, A.tocoo().col])
    dat = torch.FloatTensor(A.tocoo().data)
    return torch.sparse.FloatTensor(idx, dat, torch.Size([A.shape[0], A.shape[1]])).to(device=dev)


def extend_triple_dict(dictionary, triplets, tail_list=True):
    for triplet in triplets:
        if tail_list:
            key = (triplet[0], triplet[2])
            value = triplet[1]
        else:
            key = (triplet[2], triplet[1])
            value = triplet[0]

        if key not in dictionary:
            dictionary[key] = [value]
        elif value not in dictionary[key]:
            dictionary[key].append(value)


class DataSampler():
    def __init__(self, params, file_path, all_data_path, nBatches=1, debug=False):
        self.params = params
        end = 20001 if debug else -1
        with open(file_path) as f:
            self.data = np.array([list(map(int, sample.split())) for sample in f.read().split('\n')[1:end]], dtype=np.int64)
        assert self.data.shape[1] == 3

        with open(all_data_path) as f:
            self.all_data = np.array([list(map(int, sample.split())) for sample in f.read().split('\n')[1:end]], dtype=np.int64)
        assert self.all_data.shape[1] == 3

        self.data_set = set(map(tuple, self.data))

        self.tail_mapping = {}
        self.head_mapping = {}
        extend_triple_dict(self.tail_mapping, self.all_data, tail_list=True)
        extend_triple_dict(self.head_mapping, self.all_data, tail_list=False)

        self.ent = self.get_ent(self.data)
        self.rel = self.get_rel(self.data)

        # Build graph
        self.adj_mat = []
        for i in range(self.params.total_rel):
            # pdb.set_trace()
            idx = np.argwhere(self.data[:, 2] == i)
            adj = sp.csr_matrix((np.ones(len(idx)) / len(idx), (self.data[:, 0][idx].squeeze(1), self.data[:, 1][idx].squeeze(1))), shape=(self.params.total_ent, self.params.total_ent))
            self.adj_mat.append(adj)
            self.adj_mat.append(adj.T)
        self.adj_mat.append(sp.identity(self.adj_mat[0].shape[0]).tocsr())  # add identity matrix

        self.adj_mat = list(map(get_torch_sparse_matrix, self.adj_mat, [self.params.device] * len(self.adj_mat)))

        self.X = torch.eye(self.params.total_ent).to(device=self.params.device)

        self.batch_size = int(len(self.data) / nBatches)

        self.idx = np.arange(len(self.data))

        logging.info('Loaded data sucessfully from %s. Samples = %d; Total entities = %d; Total relations = %d' % (file_path, len(self.data), len(self.ent), len(self.rel)))

    def get_ent(self, debug=False):
        return set([i.item() for i in self.data[:, 0:2].reshape(-1)])

    def get_rel(self, debug=False):
        return set([i.item() for i in self.data[:, 2]])

    def get_batch(self, n_batch):
        if n_batch == 0:
            np.random.shuffle(self.idx)
        # pdb.set_trace()
        ids = self.idx[n_batch * self.batch_size: (n_batch + 1) * self.batch_size]
        pos_batch = self.data[ids]

        neg_batch = self.get_negative_batch(pos_batch)

        batch = np.concatenate((pos_batch, neg_batch), axis=0)

        return batch[:, 0], batch[:, 1], batch[:, 2]

    def _sample_negative(self, sample):
        neg_sample = np.array(sample)
        if random.random() < 0.5:
            while tuple(neg_sample) in self.data_set:
                neg_sample[0] = np.random.randint(0, len(self.ent))
        else:
            while tuple(neg_sample) in self.data_set:
                neg_sample[1] = np.random.randint(0, len(self.ent))
        return neg_sample

    def get_negative_batch(self, batch):
        neg_batch = np.zeros(batch.shape, dtype=np.int64)
        for i, sample in enumerate(batch):
            neg_batch[i] = self._sample_negative(sample)

        return neg_batch
