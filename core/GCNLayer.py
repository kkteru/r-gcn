import pdb
import torch
from torch import nn


class GCNLayer(nn.Module):
    def __init__(self, params, in_size, out_size, bias=False):
        super(GCNLayer, self).__init__()
        self.params = params
        self.in_size = in_size
        self.out_size = out_size

        self.basis_weights = nn.Parameter(torch.FloatTensor(self.params.n_basis, self.in_size, self.out_size))
        self.basis_coeff = nn.Parameter(torch.FloatTensor(self.params.total_rel, self.params.n_basis))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.params.total_rel, self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.basis_weights.data)
        nn.init.xavier_uniform_(self.basis_coeff.data)

        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data)

    def forward(self, inp, adj_mat_list):
        '''
        inp: (|E| x in_size)
        adj_mat_list: (R x |E| x |E|)
        '''

        # Aggregation (no explicit separation of Concat step here since we are simply averaging over all)
        rel_weights = torch.einsum('rb, bio -> rio', (self.basis_coeff, self.basis_weights))
        out = torch.zeros(self.params.total_ent, self.out_size).to(device=self.params.device)
        # pdb.set_trace()
        for i, mat in enumerate(adj_mat_list):
            # pdb.set_trace()
            if inp is not None:
                emb_acc = torch.sparse.mm(mat, inp)  # (|E| x in_size)
            else:
                emb_acc = mat
            out += torch.matmul(emb_acc, rel_weights[i])
            if self.bias is not None:
                out += self.bias[i].unsqueeze(0)  # (|E| x out_size)

        return out  # (|E| x out_size)
