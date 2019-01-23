import os
import logging

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

        self.model_params = list(self.encoder.parameters()) + (list(self.decoder.parameters()) if decoder is not None else list(self.classifier.parameters()))
        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model_params, lr=params.lr, momentum=params.momentum)

    def classifier_one_step(self):
        train_batch = self.classifier_data['train_idx']  # (batch_size)
        y = self.classifier_data['y']  # y: (batch_size, n)
        adj_mat = self.classifier_data['A']
        y = torch.LongTensor(np.array(np.argmax(y[train_batch], axis=-1)).squeeze())  # y: (batch_size)

        # print(self.encoder.rel_trans)
        ent_emb = self.encoder(adj_mat)
        scores = self.classifier(ent_emb, train_batch)
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

        ent_emb = self.encoder(adj_mat)
        # print('done with encoding')
        score = self.decoder(batch_h, batch_t, batch_r, ent_emb)
        # print('done with decoding')

        y = torch.ones(len(score))
        y[int(len(score) / 2): len(score)] = 0

        loss = F.binary_cross_entropy(score, y, reduction='sum')
        # print('Loss calculates')
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
