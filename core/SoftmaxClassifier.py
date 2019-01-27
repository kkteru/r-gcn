import torch
import torch.nn as nn
import torch.nn.functional as F

# n : n_class


class SoftmaxClassifier(nn.Module):
    def __init__(self, params):
        super(SoftmaxClassifier, self).__init__()
        self.params = params
        self.weights = nn.Parameter(torch.rand(self.params.emb_dim, self.params.n_class, requires_grad=True))  # (d, n)

    def get_prediction(self, ent_emb, batch):
        scores = torch.matmul(ent_emb[batch], self.weights)  # (batch_size, n)
        pred = torch.argmax(scores, dim=-1)

        return pred

    def forward(self, ent_emb, batch):
        '''
        ent_emb: (N, d)
        batch: (batch_size) Node indices which we need the softmax score for.
        '''
        scores = torch.matmul(ent_emb[batch], self.weights)  # (batch_size, n)

        return scores  # F.log_softmax(scores, dim=1)
