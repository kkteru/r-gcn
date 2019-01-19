import torch
import torch.nn as nn
import torch.nn.functional as F

# R: Total number of relations
# N: Total number of entities (nodes)


class GCN(nn.Module):
    def __init__(self, params):
        '''
        adj_mat: (R + 1, N, N)
        '''
        super(GCN, self).__init__()
        self.params = params
        self.n_layers = self.params.gcn_layers
        self.ent_emb = torch.rand(self.params.total_ent, self.params.emb_dim)  # (N, d)
        self.rel_trans = nn.Parameter(torch.rand(self.params.total_rel, self.params.emb_dim, self.params.emb_dim, requires_grad=True))  # (R + 1 x d x d) + 1 for the self loop

        def forward(self, adj_mat):
            adj_mat = adj_mat / torch.sum(adj_mat, dim=1, keepdim=True)  # (R + 1 X N X N)
            for l in self.n_layers:
                emb_acc = torch.matmul(adj_mat, self.ent_emb)  # (R + 1 X N X d)
                tmp = torch.matmul(self.rel_trans, emb_acc.transpose(1, 2)).transpose(1, 2)  # (R + 1 X N X d)
                self.ent_emb = F.relu(torch.sum(tmp, dim=0))  # (N x d)

            return self.ent_emb
