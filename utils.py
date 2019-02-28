import os
import argparse
import logging
import json
import torch
# import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import pdb

from core import SoftmaxClassifier, GCN, EmbLookUp

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(MAIN_DIR, 'data/FB15K237')
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train2id.txt')
VALID_DATA_PATH = os.path.join(DATA_PATH, 'valid2id.txt')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test2id.txt')


def csr_zero_rows(csr, rows_to_zero):
    """Set rows given by rows_to_zero in a sparse csr matrix to zero.
    NOTE: Inplace operation! Does not return a copy of sparse matrix."""
    rows, cols = csr.shape
    mask = np.ones((rows,), dtype=np.bool)
    mask[rows_to_zero] = False
    nnz_per_row = np.diff(csr.indptr)

    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[rows_to_zero] = 0
    csr.data = csr.data[mask]
    csr.indices = csr.indices[mask]
    csr.indptr[1:] = np.cumsum(nnz_per_row)
    csr.eliminate_zeros()
    return csr


def csc_zero_cols(csc, cols_to_zero):
    """Set rows given by cols_to_zero in a sparse csc matrix to zero.
    NOTE: Inplace operation! Does not return a copy of sparse matrix."""
    rows, cols = csc.shape
    mask = np.ones((cols,), dtype=np.bool)
    mask[cols_to_zero] = False
    nnz_per_row = np.diff(csc.indptr)

    mask = np.repeat(mask, nnz_per_row)
    nnz_per_row[cols_to_zero] = 0
    csc.data = csc.data[mask]
    csc.indices = csc.indices[mask]
    csc.indptr[1:] = np.cumsum(nnz_per_row)
    csc.eliminate_zeros()
    return csc


def sp_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (dim, 1)
    data = np.ones(len(idx_list))
    row_ind = list(idx_list)
    col_ind = np.zeros(len(idx_list))
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors."""
    sp_nodes = sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(sp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def bfs(adj, roots):
    """
    Perform BFS on a graph given by an adjaceny matrix adj.
    Can take a set of multiple root nodes.
    Root nodes have level 0, first-order neighors have level 1, and so on.]
    """
    visited = set()
    current_lvl = set(roots)
    while current_lvl:
        for v in current_lvl:
            visited.add(v)

        next_lvl = get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference
        yield next_lvl

        current_lvl = next_lvl


def bfs_relational(adj_list, roots):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = list()
    for rel in range(len(adj_list)):
        next_lvl.append(set())

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        for rel in range(len(adj_list)):
            next_lvl[rel] = get_neighbors(adj_list[rel], current_lvl)
            next_lvl[rel] -= visited  # set difference

        yield next_lvl

        current_lvl = set.union(*next_lvl)


def bfs_sample(adj, roots, max_lvl_size):
    """
    BFS with node dropout. Only keeps random subset of nodes per level up to max_lvl_size.
    'roots' should be a mini-batch of nodes (set of node indices).

    NOTE: In this implementation, not every node in the mini-batch is guaranteed to have
    the same number of neighbors, as we're sampling for the whole batch at the same time.
    """
    visited = set(roots)
    current_lvl = set(roots)
    while current_lvl:

        next_lvl = get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        for v in next_lvl:
            visited.add(v)

        yield next_lvl

        current_lvl = next_lvl


def get_splits(y, train_idx, test_idx, validation=True):
    # Make dataset splits
    # np.random.shuffle(train_idx)
    if validation:
        idx_train = train_idx[len(train_idx) / 5:]
        idx_val = train_idx[:len(train_idx) / 5]
        idx_test = idx_val  # report final score on validation set for hyperparameter optimization
    else:
        idx_train = train_idx
        idx_val = train_idx  # no validation
        idx_test = test_idx

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)

    y_train[idx_train] = np.array(y[idx_train].todense())
    y_val[idx_val] = np.array(y[idx_val].todense())
    y_test[idx_test] = np.array(y[idx_test].todense())

    return y_train, y_val, y_test, idx_train, idx_val, idx_test


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten())
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten())
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def binary_crossentropy(preds, labels):
    return np.mean(-labels * np.log(preds) - (1 - labels) * np.log(1 - preds))


def two_class_accuracy(preds, labels, threshold=0.5):
    return np.mean(np.equal(labels, preds > 0.5))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def evaluate_preds_sigmoid(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(binary_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(two_class_accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag. use 0 or 1")


def initialize_experiment(params):

    exps_dir = os.path.join(MAIN_DIR, 'experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    params.exp_dir = os.path.join(exps_dir, params.experiment_name)

    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)

    file_handler = logging.FileHandler(os.path.join(params.exp_dir, "log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
        json.dump(vars(params), fout)


def initialize_model(params, classifier_data, fresh=True):

    if not fresh and os.path.exists(os.path.join(params.exp_dir, 'best_gcn.pth')):
        logging.info('Loading existing model from %s' % os.path.join(params.exp_dir, 'best_gcn.pth'))
        enc = torch.load(os.path.join(params.exp_dir, 'best_gcn.pth')).to(device=params.device)  # Update these
        logging.info('Loading existing model from %s' % os.path.join(params.exp_dir, 'best_classifier.pth'))
        sm_classifier = torch.load(os.path.join(params.exp_dir, 'best_classifier.pth')).to(device=params.device)  # Update these
    else:
        logging.info('No existing model found. Initializing new model..')
        if params.no_encoder:
            enc = EmbLookUp(params, params.total_ent).to(device=params.device)
        else:
            enc = GCN(params).to(device=params.device)
        sm_classifier = SoftmaxClassifier(params, classifier_data['y'].shape[1]).to(device=params.device)

    return enc, sm_classifier


# class Logger(object):

#     def __init__(self, log_dir):
#         """Create a summary writer logging to log_dir."""
#         self.writer = tf.summary.FileWriter(log_dir)

#     def scalar_summary(self, tag, value, step):
#         """Log a scalar variable."""
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
#         self.writer.add_summary(summary, step)

#     def image_summary(self, tag, images, step):
#         """Log a list of images."""

#         img_summaries = []
#         for i, img in enumerate(images):
#             s = BytesIO()
#             scipy.misc.toimage(img).save(s, format="png")

#             # Create an Image object
#             img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
#                                        height=img.shape[0],
#                                        width=img.shape[1])
#             # Create a Summary value
#             img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

#         # Create and write Summary
#         summary = tf.Summary(value=img_summaries)
#         self.writer.add_summary(summary, step)

#     def histo_summary(self, tag, values, step, bins=1000):
#         """Log a histogram of the tensor of values."""

#         # Create a histogram using numpy
#         counts, bin_edges = np.histogram(values, bins=bins)

#         # Fill the fields of the histogram proto
#         hist = tf.HistogramProto()
#         hist.min = float(np.min(values))
#         hist.max = float(np.max(values))
#         hist.num = int(np.prod(values.shape))
#         hist.sum = float(np.sum(values))
#         hist.sum_squares = float(np.sum(values**2))

#         # Drop the start of the first bin
#         bin_edges = bin_edges[1:]

#         # Add bin edges and counts
#         for edge in bin_edges:
#             hist.bucket_limit.append(edge)
#         for c in counts:
#             hist.bucket.append(c)

#         # Create and write Summary
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
#         self.writer.add_summary(summary, step)
#         self.writer.flush()


def get_torch_sparse_matrix(A, dev):
    '''
    A : list of sparse adjacency matrices
    '''
    idx = torch.LongTensor([A.tocoo().row, A.tocoo().col])
    dat = torch.FloatTensor(A.tocoo().data)
    return torch.sparse.FloatTensor(idx, dat, torch.Size([A.shape[0], A.shape[1]])).to(device=dev)
