import torch
import torch.nn as nn


class DistMult(nn.Module):
    def __init__(self, params):
        super(DistMult, self).__init__()
        self.params = params
        self.rel_emb = nn.Parameter(torch.empty(self.params.total_rel, self.params.emb_dim), requires_grad=True)  # (R_, d) R_ is just the relations without the direction and self connection
        nn.init.xavier_uniform_(self.rel_emb.data)

    def forward(self, head_emb, tail_emb, batch_rel):
        '''
        head_emb : (batch_size, d)
        tail_emb : (batch_size, d)
        batch_rel : (batch_size)
        ent_emb : (N, d)
        '''
        rels = self.rel_emb[batch_rel]  # (batch_size, d)

        score = torch.sigmoid(torch.matmul(torch.matmul(head_emb.unsqueeze(-1).transpose(1, 2), torch.diag_embed(rels)), tail_emb.unsqueeze(-1))).squeeze()  # (batch_size)

        return score
