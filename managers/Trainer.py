import os
import logging
import pdb

import torch
import torch.optim as optim
import torch.nn as nn


class Trainer():
    def __init__(self, params, encoder, decoder, link_data_sampler):
        self.params = params
        self.encoder = encoder
        self.decoder = decoder

        self.link_data_sampler = link_data_sampler

        self.best_mr = 1e10
        self.last_mr = 1e10
        self.bad_count = 0

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')

        self.model_params = list(self.encoder.parameters()) + (list(self.decoder.parameters()) if decoder is not None else list(self.classifier.parameters()))
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), self.model_params)))
        # pdb.set_trace()
        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model_params, lr=params.lr, momentum=params.momentum)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model_params, lr=params.lr)

    def link_pred_one_step(self, n_batch):
        '''
        n_batch: scalar value
        '''
        batch_h, batch_t, batch_r = self.link_data_sampler.get_batch(n_batch)
        adj_mat = self.link_data_sampler.adj_mat
        X = self.link_data_sampler.X

        ent_emb = self.encoder(X, adj_mat)

        head_emb = ent_emb[batch_h]
        tail_emb = ent_emb[batch_t]
        score = self.decoder(head_emb, tail_emb, batch_r)

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

        return loss

    def save_link_predictor(self, log_data):
        if log_data['mr'] < self.best_mr:
            torch.save(self.encoder, os.path.join(self.params.exp_dir, 'best_gcn.pth'))  # Does it overwrite or fuck with the existing file?
            torch.save(self.decoder, os.path.join(self.params.exp_dir, 'best_distmul.pth'))  # Does it overwrite or fuck with the existing file?
            logging.info('Better models found w.r.t MR. Saved it!')
            self.best_mr = log_data['mr']
        else:
            self.bad_count = self.bad_count + 1
            if self.bad_count > self.params.patience:
                logging.info('Darn it! I dont have any more patience to give this model.')
                return False
        self.last_mr = log_data['mr']
        return True
