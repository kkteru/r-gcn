import os
import logging
import pdb


import numpy as np
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

        self.best_metric = 0
        self.last_metric = 0
        self.bad_count = 0

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')

        self.model_params = list(self.encoder.parameters()) + (list(self.decoder.parameters()) if decoder is not None else list(self.classifier.parameters()))
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), self.model_params)))
        # pdb.set_trace()
        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model_params, lr=params.lr, momentum=params.momentum)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model_params, lr=params.lr)

    def classifier_one_step(self):
        train_batch = self.classifier_data['train_idx']  # (batch_size)
        y = self.classifier_data['y']  # y: (batch_size, n)

        # pdb.set_trace()

        adj_mat = self.classifier_data['A']

        if self.params.dataset == 'cora':
            y = y[train_batch].to(device=self.params.device)
        else:
            y = torch.LongTensor(np.array(np.argmax(y[train_batch], axis=-1)).squeeze()).to(device=self.params.device)  # y: (batch_size)

        # pdb.set_trace()
        X = torch.Tensor(self.classifier_data['feat']).to(device=self.params.device)
        # pdb.set_trace()
        # print(self.encoder.rel_trans)
        ent_emb = self.encoder(X, adj_mat)
        # scores = ent_emb[train_batch]
        scores = self.classifier(ent_emb, train_batch)

        # pdb.set_trace()

        loss = F.cross_entropy(scores, y)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model_params, self.params.clip)
        self.optimizer.step()

        return loss

    def link_pred_one_step(self, batch_size):
        '''
        batch_size: scalar value
        '''
        batch_h, batch_t, batch_r = self.link_data_sampler.get_batch(batch_size)
        adj_mat = self.link_data_sampler.adj_mat
        X = self.link_data_sampler.X

        ent_emb = self.encoder(X, adj_mat)
        score = self.decoder(batch_h, batch_t, batch_r, ent_emb)

        pos_score = score[0: int(len(score) / 2)]
        neg_score = score[int(len(score) / 2): len(score)]

        # y = torch.ones(len(score))
        # y[int(len(score) / 2): len(score)] = 0

        # loss = F.binary_cross_entropy(score, y, reduction='sum')
        loss = self.criterion(pos_score, neg_score, torch.Tensor([-1]).to(device=self.params.device))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model_params, self.params.clip)
        self.optimizer.step()

        # print(self.decoder.rel_emb)

        return loss

    def save_link_predictor(self, log_data):
        if log_data['mrr'] > self.last_metric:
            self.bad_count = 0

            if log_data['mrr'] > self.best_metric:
                torch.save(self.encoder, os.path.join(self.params.exp_dir, 'best_gcn.pth'))  # Does it overwrite or fuck with the existing file?
                torch.save(self.decoder, os.path.join(self.params.exp_dir, 'best_distmul.pth'))  # Does it overwrite or fuck with the existing file?
                logging.info('Better models found w.r.t MRR. Saved it!')
                self.best_metric = log_data['mrr']
        else:
            self.bad_count = self.bad_count + 1
            if self.bad_count > self.params.patience:
                logging.info('Darn it! I dont have any more patience to give this model.')
                return False
        self.last_metric = log_data['mrr']
        return True

    def save_classifier(self, log_data):
        if log_data['acc'] > self.last_metric:
            self.bad_count = 0

            if log_data['acc'] > self.best_metric:
                torch.save(self.encoder, os.path.join(self.params.exp_dir, 'best_gcn.pth'))  # Does it overwrite or fuck with the existing file?
                torch.save(self.classifier, os.path.join(self.params.exp_dir, 'best_classifier.pth'))  # Does it overwrite or fuck with the existing file?
                logging.info('Better models found w.r.t accuracy. Saved it!')
                self.best_metric = log_data['acc']
        else:
            self.bad_count = self.bad_count + 1
            if self.bad_count > self.params.patience:
                logging.info('Darn it! I dont have any more patience to give this model.')
                return False
        self.last_metric = log_data['acc']
        return True
