import torch
from torch import nn


class GCNLayer(nn.Module):
    def __init__(self, params, in_size, out_size, n_rel, bias=True):
        super(GCNLayer, self).__init__()
        self.params = params
        self.in_size = in_size
        self.out_size = out_size
        self.n_rel = n_rel

        self.weights = nn.Parameter(torch.FloatTensor(n_rel, self.in_size, self.out_size))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(n_rel, self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)

        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data)

    def forward(self, inp, adj_mat_list):
        '''
        inp: (|E| x in_size)
        adj_mat_list: (R x |E| x |E|)
        '''

        # Aggregation (no explicit separation of Concat step here since we are simply averaging over all)
        emb_acc = torch.empty(self.n_rel, inp.shape[0], inp.shape[1]).to(device=self.params.device)
        for i, mat in enumerate(adj_mat_list):
            emb_acc[i] = torch.sparse.mm(mat, inp).to(device=self.params.device)

        out_array = torch.matmul(emb_acc, self.weights)  # (R x |E| x out_size)

        if self.bias is not None:
            out_array = out_array + self.bias.unsqueeze(1)

        out = torch.sum(out_array, dim=0)  # Could this aggregation over relations be done better?

        return out  # (|E| x out_size)
