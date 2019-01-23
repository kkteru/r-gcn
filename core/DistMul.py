import torch
import torch.nn as nn


class DistMul(nn.Module):
    def __init__(self, params):
        super(DistMul, self).__init__()
        self.params = params
        self.rel_emb = nn.Parameter(torch.rand((self.params.total_rel - 1) // 2, self.params.emb_dim), requires_grad=True)  # (R_, d, d) R_ is just the relations without the direction and self connection

    def get_score(self, heads, tails, rels):
        '''
        heads : (batch/sample_size, d)
        tails : (batch/sample_size, d)
        rels : (batch/sample_size, d, d)
        '''

        # print('calculating decoder scores...')
        score = torch.sigmoid(torch.matmul(torch.matmul(heads.unsqueeze(-1).transpose(1, 2), torch.diag_embed(rels)), tails.unsqueeze(-1))).squeeze()

        return score

    def forward(self, batch_h, batch_t, batch_r, ent_emb):
        '''
        batch_h : (batch_size)
        batch_r : (batch_size)
        batch_t : (batch_size)
        ent_emb : (N, d)
        '''
        heads = ent_emb[batch_h]  # (batch_size, d)
        tails = ent_emb[batch_t]  # (batch_size, d)
        rels = self.rel_emb[batch_r]  # (batch_size, d, d)

        return self.get_score(heads, tails, rels)
