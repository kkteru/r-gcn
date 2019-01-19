import torch
import torch.nn as nn

# n : n_class


class SoftmaxClassifier(nn.Module):
    def __init__(self, params):
        super(SoftmaxClassifier, self).__init__()
        self.params = params
        self.weights = nn.Parameter(torch.rand(self.params.emb_dim, self.params.n_class, requires_grad=True))  # (d, n)

    def forward(self, ent_emb, batch):
        '''
        ent_emb: (N, d)
        batch: (batch_size) Node indices which we need the softmax score for.
        '''
        scores = torch.matmul(ent_emb[batch], self.weights)  # (batch_size, n)

        return scores
