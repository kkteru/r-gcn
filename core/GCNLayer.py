import pdb
import torch
from torch import nn


class GCNLayer(nn.Module):
    def __init__(self, params, in_size, out_size, bias=False):
        super(GCNLayer, self).__init__()
        self.params = params
        self.in_size = in_size
        self.out_size = out_size

        if self.params.n_basis > 0:
            self.basis_weights = nn.Parameter(torch.FloatTensor(self.params.n_basis, self.in_size, self.out_size))
            self.basis_coeff = nn.Parameter(torch.FloatTensor(self.params.total_rel, self.params.n_basis))
        else:
            self.weights = nn.Parameter(torch.FloatTensor(self.params.total_rel, self.in_size, self.out_size))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.params.n_basis > 0:
            nn.init.xavier_uniform_(self.basis_weights.data)
            nn.init.xavier_uniform_(self.basis_coeff.data)
        else:
            nn.init.xavier_uniform_(self.weights.data)

        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data)

    def forward(self, inp, adj_mat_list):
        '''
        inp: (|E| x in_size)
        adj_mat_list: (R x |E| x |E|)
        '''

        # Aggregation (no explicit separation of Concat step here since we are simply averaging over all)
        if self.params.n_basis > 0:
            rel_weights = torch.einsum('rb, bio -> rio', (self.basis_coeff, self.basis_weights))
        else:
            rel_weights = self.weights

        weights = rel_weights.view(rel_weights.shape[0] * rel_weights.shape[1], rel_weights.shape[2])  # (in_size * R, out_size)

        emb_acc = []

        if inp is not None:
            for mat in adj_mat_list:
                emb_acc.append(torch.sparse.mm(mat, inp))  # (|E| x in_size)
        else:
            emb_acc = adj_mat_list

        tmp = torch.cat(emb_acc, dim=1)  # (|E|, in_size * R)

        # HORRIBLE HACK!
        while torch.isnan(torch.norm(tmp)):
            emb_acc = []

            if inp is not None:
                for i, mat in enumerate(adj_mat_list):
                    emb_acc.append(torch.sparse.mm(mat, inp))  # (|E| x in_size)
            else:
                emb_acc = adj_mat_list

            tmp = torch.cat(emb_acc, dim=1)  # (|E|, in_size * R)

        out = torch.matmul(tmp, weights)  # (|E| x out_size)

        if self.bias is not None:
            out += self.bias.unsqueeze(0)  # (|E| x out_size)

        return out  # (|E| x out_size)
