import torch
import torch.nn as nn


class DistMult(nn.Module):
    def __init__(self, params):
        super(DistMult, self).__init__()
        self.params = params
        self.rel_emb = nn.Parameter(torch.empty(self.params.total_rel, self.params.emb_dim), requires_grad=True)  # (R, d)
        nn.init.xavier_uniform_(self.rel_emb.data)

    def get_all_scores(self, ent_dict, batch_h, batch_t, batch_rel, mode='head'):
        '''
        ent_dict: All entity embeddings (|E| x d)
        batch_h: Head entity indices (B x 1)
        batch_t: Tail entity indices (B x 1)
        batch_rel: Relation indices (B x 1)
        '''

        h_e = ent_dict[batch_h]  # (B x d)
        t_e = ent_dict[batch_t]  # (B x d)
        r_e = self.rel_emb[batch_h]  # (B x d)

        if mode == 'head':
            c_e = torch.matmul(torch.diag_embed(r_e), t_e.unsqueeze(-1)).squeeze()  # (B x d)

        if mode == 'tail':
            c_e = torch.matmul(h_e.unsqueeze(-2), torch.diag_embed(r_e)).squeeze()  # (B x d)

        distList = torch.matmul(c_e, ent_dict.transpose(0, 1))  # (B x |E|)

        rankList = torch.argsort(distList, dim=1)

        return rankList

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
