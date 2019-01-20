import numpy as np


class Evaluator():
    def __init__(self, encoder, decoder, classifier, classification_data):
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.classification_data = classification_data

    def classifier_log_data(self):
        valid_idx = self.classification_data['valid_idx']
        pred = self.classifier.get_prediction(self.encoder.ent_emb, valid_idx)
        acc = np.mean(pred.numpy() == np.argmax(self.classification_data['y'][valid_idx], axis=1))

        log_data = dict([
            ('acc', acc)])

        return log_data
