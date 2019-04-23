import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from .GCNLayer import GCNLayer
# R: Total number of relations
# N: Total number of entities (nodes)


class GCN(nn.Module):
    def __init__(self, params, layer_sizes=None, inp=None):
        super(GCN, self).__init__()

        self.params = params
        self.layer_sizes = [self.params.emb_dim] * self.params.gcn_layers if layer_sizes is None else layer_sizes

        assert len(self.layer_sizes) == params.gcn_layers

        if inp is None:
            self.node_init = nn.Parameter(torch.FloatTensor(params.total_ent, params.emb_dim))
            nn.init.xavier_uniform_(self.node_init.data)
        else:
            self.node_init = inp

        self.layers = nn.ModuleList()

        _l = self.node_init.shape[1]

        for l in self.layer_sizes:
            self.layers.append(GCNLayer(params, _l, l, 2 * self.params.total_rel + 1))
            _l = l
        self.ent_emb = None

    def forward(self, adj_mat_list):
        '''
        inp: (|E| x d)
        adj_mat_list: (R x |E| x |E|)
        '''
        out = self.node_init
        for layer in self.layers:
            out = layer(out, adj_mat_list)
            out = F.relu(out)  # Applies to output layer too!
            out = F.normalize(out)
        self.ent_emb = out
        return out
