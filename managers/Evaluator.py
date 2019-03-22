import numpy as np
import torch
import pdb


class Evaluator():
    def __init__(self, params, encoder, classifier, classification_data):
        self.encoder = encoder
        self.classifier = classifier
        self.classification_data = classification_data
        self.params = params

    def classifier_log_data(self, data='valid'):
        if data == 'valid':
            idx = self.classification_data['train_idx']
        elif data == 'test':
            idx = self.classification_data['test_idx']

        # pdb.set_trace()
        ent_emb = self.encoder.ent_emb
        pred = ent_emb[idx]
        # pred = self.classifier(ent_emb[idx])

        acc = np.mean(np.argmax(pred.detach().cpu().numpy(), axis=1) == np.argmax(self.classification_data['y'][idx], axis=1).squeeze())

        log_data = dict([
            ('acc', acc)])

        return log_data
