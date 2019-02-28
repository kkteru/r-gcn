import torch
from torch import nn
import torch.nn.functional as F


class EmbLookUp(nn.Module):
    def __init__(self, params, in_size):
        super(EmbLookUp, self).__init__()
        self.params = params

        self.ent_emb = nn.Parameter(torch.FloatTensor(in_size, self.params.emb_dim))  # (N, d)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ent_emb.data)

    def forward(self, t1):
        # self.ent_emb = F.normalize(self.ent_emb)

        return self.ent_emb
