import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class Trainer():
    def __init__(self, params, encoder, decoder, classifier, classifier_data, link_data_sampler):
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

        self.classifier_data = classifier_data
        self.link_data_sampler = link_data_sampler

        model_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.classifier.parameters())
        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum)

        self.criterion = nn.BCELoss()

    def classifier_one_step(self):
        train_idx = self.classifier_data['train_idx']  # (batch_size)
        y = self.classifier_data['y']  # y: (batch_size, n)
        adj_mat = self.classifier_data['A']
        y = torch.max(y, dim=-1)  # y: (batch_size)

        ent_emb = self.encoder(adj_mat)
        scores = self.classifier(ent_emb, train_idx)
        loss = F.cross_entropy(scores, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def link_pred_one_step(self, batch_size):
        '''
        batch_size: scalar value
        '''
        batch_h, batch_t, batch_r = self.link_data_sampler.get_batch(batch_size)
        adj_mat = None  # GET THIS!

        ent_emb = self.encoder(adj_mat)
        score = self.decoder(batch_h, batch_t, batch_r, ent_emb)

        y = torch.ones(len(score))
        y[int(len(score) / 2): len(score)] = 0

        loss = self.criterion(score, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
