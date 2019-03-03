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
        self.layer_sizes = [self.params.emb_dim] * (self.params.gcn_layers - 1) + [self.params.n_class] if layer_sizes is None else layer_sizes

        assert len(self.layer_sizes) == params.gcn_layers

        self.node_init = None
        _l = self.params.total_ent
        if inp is not None:
            self.node_init = torch.FloatTensor(inp).to(device=params.device)
            _l = self.node_init.shape[1]

        self.layers = nn.ModuleList()

        for l in self.layer_sizes:
            self.layers.append(GCNLayer(params, _l, l))
            _l = l
        self.ent_emb = None

    def forward(self, adj_mat_list):
        '''
        inp: (|E| x d)
        adj_mat_list: (R x |E| x |E|)
        '''
        out = self.node_init
        for i, layer in enumerate(self.layers):
            if i != 0:
                out = F.relu(out)
            out = layer(out, adj_mat_list)
            # out = F.normalize(out)
        self.ent_emb = out
        return out
