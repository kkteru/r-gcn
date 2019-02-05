import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
# R: Total number of relations
# N: Total number of entities (nodes)


class GCN(nn.Module):
    def __init__(self, params):
        super(GCN, self).__init__()
        self.params = params
        self.n_layers = self.params.gcn_layers
        self.ent_emb = nn.Parameter(torch.empty(self.params.feat_in, self.params.emb_dim), requires_grad=True)  # (N, d)
        self.final_emb = None
        if not self.params.no_encoder:
            self.rel_trans = nn.Parameter(torch.empty(self.n_layers, self.params.total_rel, self.params.emb_dim, self.params.emb_dim), requires_grad=True)  # (R + 1 x d x d); + 1 for the self loop
            nn.init.xavier_uniform_(self.rel_trans.data)
        nn.init.xavier_uniform_(self.ent_emb.data)

    def forward(self, x, adj_mat):
        '''
        A : list of sparse torch adjacency matrices
        '''
        emb = self.ent_emb
        emb = torch.matmul(x, emb)
        if not self.params.no_encoder:
            emb_acc = torch.empty(self.params.total_rel, self.params.total_ent, self.params.emb_dim).to(device=self.params.device)  # (R + 1 X N X d)
            for l in range(self.n_layers):
                # pdb.set_trace()
                for i, mat in enumerate(adj_mat):
                    # pdb.set_trace()
                    emb_acc[i] = torch.sparse.mm(mat, emb).to(device=self.params.device)
                # pdb.set_trace()
                tmp = torch.matmul(self.rel_trans[l], emb_acc.transpose(1, 2)).transpose(1, 2)  # (R + 1 X N X d) Shoud be different weights for different layers?
                emb = F.relu(torch.sum(tmp, dim=0))
            emb = F.normalize(emb)
        self.final_emb = emb
        return emb
