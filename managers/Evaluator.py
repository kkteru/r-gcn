import numpy as np


class Evaluator():
    def __init__(self, encoder, decoder, classifier, classification_data, link_data_sampler, neg_sample_size=0):
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.classification_data = classification_data
        self.link_data_sampler = link_data_sampler
        self.neg_sample_size = neg_sample_size if neg_sample_size != 0 else len(link_data_sampler.data)

    def classifier_log_data(self):
        valid_idx = self.classification_data['train_idx']
        pred = self.classifier.get_prediction(self.encoder.ent_emb, valid_idx)
        acc = np.mean(pred.numpy() == np.argmax(self.classification_data['y'][valid_idx], axis=1))

        log_data = dict([
            ('acc', acc)])

        return log_data

    def _rank_triplets(self, sample):
        idx = np.random.random_integers(0, len(self.link_data_sampler.ent) - 1, self.neg_sample_size)

        head_ids = np.array(list(self.link_data_sampler.ent))[idx]
        head_ids[0] = sample[0]

        heads = self.encoder.ent_emb[head_ids]  # (sample_size, d)
        tails = self.encoder.ent_emb[[sample[1]] * len(head_ids)]  # (sample_size, d)
        rels = self.decoder.rel_emb[[sample[2]] * len(head_ids)]  # (sample_size, d, d)

        # print(heads.shape, tails.shape, rels.shape)

        scores = self.decoder.get_score(heads, tails, rels)

        assert scores.shape == (len(head_ids), )

        return np.where(head_ids[scores.argsort()] == sample[0])[0][0] + 1

    def link_log_data(self):
        ranks = np.array(list(map(self._rank_triplets, self.link_data_sampler.data)))

        assert len(ranks) == len(self.link_data_sampler.data)

        hit10 = np.sum(ranks < 10) / len(ranks)
        mrr = np.mean(1 / ranks)
        mr = np.mean(ranks)

        log_data = dict([
            ('hit@10', hit10),
            ('mrr', mrr),
            ('mr', mr)])

        return log_data
