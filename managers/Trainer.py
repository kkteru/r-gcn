import os
import logging
import pdb


import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class Trainer():
    def __init__(self, params, encoder, classifier, classifier_data):
        self.params = params
        self.encoder = encoder
        self.classifier = classifier

        self.classifier_data = classifier_data

        self.best_metric = 0
        self.last_metric = 0
        self.bad_count = 0

        self.model_params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), self.model_params)))
        # pdb.set_trace()
        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model_params, lr=params.lr, weight_decay=self.params.l2)

    def classifier_one_step(self):
        train_batch = self.classifier_data['train_idx']  # (batch_size)
        y = self.classifier_data['y'][train_batch]  # y: (batch_size, n)
        y = torch.LongTensor(np.array(np.argmax(y, axis=-1)).squeeze()).to(device=self.params.device)  # y: (batch_size)

        # pdb.set_trace()

        adj_mat = self.classifier_data['A']

        # pdb.set_trace()
        ent_emb = self.encoder(adj_mat)
        scores = ent_emb[train_batch]
        # scores = self.classifier(ent_emb[train_batch])

        # pdb.set_trace()
        loss = F.cross_entropy(scores, y)

        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.model_params, self.params.clip)
        self.optimizer.step()

        return loss

    def save_classifier(self, log_data):
        if log_data['acc'] >= self.best_metric:
            torch.save(self.encoder, os.path.join(self.params.exp_dir, 'best_gcn.pth'))  # Does it overwrite or fuck with the existing file?
            torch.save(self.classifier, os.path.join(self.params.exp_dir, 'best_classifier.pth'))  # Does it overwrite or fuck with the existing file?
            logging.info('Better models found w.r.t accuracy. Saved it!')
            self.best_metric = log_data['acc']
            self.bad_count = 0
        else:
            self.bad_count = self.bad_count + 1
            if self.bad_count > self.params.patience:
                logging.info('Darn it! I dont have any more patience to give this model.')
                return False
        self.last_metric = log_data['acc']
        return True
