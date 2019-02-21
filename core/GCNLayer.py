import torch
from torch import nn


class GCNLayer(nn.Module):
    def __init__(self, params, in_size, out_size, n_rel, bias=True):
        super(GCNLayer, self).__init__()
        self.params = params
        self.in_size = in_size
        self.out_size = out_size
        self.n_rel = n_rel

        # self.weights = nn.Parameter(torch.FloatTensor(n_rel, self.in_size, self.out_size))
        self.basis_weights = nn.Parameter(torch.FloatTensor(self.params.n_basis, self.in_size, self.out_size))
        self.basis_coeff = nn.Parameter(torch.FloatTensor(n_rel, self.params.n_basis))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(n_rel, self.out_size))
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
        out = torch.zeros(inp.shape[0], self.out_size)
        for i, mat in enumerate(adj_mat_list):
            emb_acc = torch.sparse.mm(mat, inp).to(device=self.params.device)  # (|E| x inp_size)
            rel_weight = torch.einsum('b, bio -> io', (self.basis_coeff[i], self.basis_weights))
            out += torch.matmul(emb_acc, rel_weight)
            if self.bias is not None:
                out += self.bias[i].unsqueeze(0)  # (|E| x out_size)

        return out  # (|E| x out_size)
