import torch
import torch.nn as nn

# n : n_class


class SoftmaxClassifier(nn.Module):
    def __init__(self, params, n_class):
        super(SoftmaxClassifier, self).__init__()
        self.params = params
        self.weights = nn.Parameter(torch.rand(self.params.emb_dim, n_class, requires_grad=True))  # (d, n)

    def forward(self, ent_emb):
        '''
        ent_emb: (batch_size, d)
        '''
        scores = torch.matmul(ent_emb, self.weights)  # (batch_size, n)

        return scores
