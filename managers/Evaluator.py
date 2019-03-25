import numpy as np
import pdb


class Evaluator():
    def __init__(self, params, encoder, decoder, link_data_sampler, neg_sample_size=0):
        self.encoder = encoder
        self.decoder = decoder

        self.link_data_sampler = link_data_sampler
        self.params = params

    def get_log_data(self, eval_mode='head'):

        mr = []
        mrr = []
        hit10 = []

        if eval_mode == 'head' or eval_mode == 'avg':

            distArrayHead = self.decoder.get_all_scores(self.encoder.ent_emb.data, self.link_data_sampler.data[:, 0],
                                                        self.link_data_sampler.data[:, 1], self.link_data_sampler.data[:, 2],
                                                        'head').cpu().numpy()

            rankListHead = [np.sum(dist < dist[n]) + 1 for (dist, n) in zip(distArrayHead, self.link_data_sampler.data[:, 0])]

            if self.params.filter:
                for i, (head, tail, rel) in enumerate(self.link_data_sampler.data):
                    if (rel, tail) in self.link_data_sampler.head_mapping:
                        heads = self.link_data_sampler.head_mapping[(rel, tail)]
                        tails = [tail] * len(heads)
                        rels = [rel] * len(heads)

                        head_emb = self.encoder.ent_emb.data[heads]
                        tail_emb = self.encoder.ent_emb.data[tails]

                        scores = self.decoder.get_score(head_emb, tail_emb, rels).cpu().numpy()
                        rankListHead[i] -= np.sum(scores < (distArrayHead[i][head] + 1e-3)) - 1

            isHit10ListHead = [x for x in rankListHead if x <= 10]

            assert len(rankListHead) == len(self.link_data_sampler.data)

            mr.append(np.mean(rankListHead))
            mrr.append(np.mean(1 / np.array(rankListHead)))
            hit10.append(len(isHit10ListHead) / len(rankListHead))

# -------------------------------------------------------------------- #

        if eval_mode == 'tail' or eval_mode == 'avg':
            distArrayTail = self.decoder.get_all_scores(self.encoder.ent_emb.data, self.link_data_sampler.data[:, 0],
                                                        self.link_data_sampler.data[:, 1], self.link_data_sampler.data[:, 2],
                                                        'tail').cpu().numpy()
            rankListTail = [np.sum(dist < dist[n]) + 1 for (dist, n) in zip(distArrayTail, self.link_data_sampler.data[:, 1])]

            if self.params.filter:
                for i, (head, tail, rel) in enumerate(self.link_data_sampler.data):
                    if (head, rel) in self.link_data_sampler.tail_mapping:
                        tails = self.link_data_sampler.tail_mapping[(head, rel)]
                        heads = [head] * len(tails)
                        rels = [rel] * len(tails)

                        head_emb = self.encoder.ent_emb.data[heads]
                        tail_emb = self.encoder.ent_emb.data[tails]

                        scores = self.decoder.get_score(head_emb, tail_emb, rels).cpu().numpy()
                        rankListTail[i] -= np.sum(scores < (distArrayTail[i][tail] + 1e-3)) - 1

            isHit10ListTail = [x for x in rankListTail if x <= 10]

            assert len(rankListTail) == len(self.link_data_sampler.data)

            mr.append(np.mean(rankListTail))
            mrr.append(np.mean(1 / np.array(rankListTail)))
            hit10.append(len(isHit10ListTail) / len(rankListTail))

        mr = np.mean(mr)
        mrr = np.mean(mrr)
        hit10 = np.mean(hit10)

        log_data = dict([
            ('hit@10', hit10),
            ('mr', mr),
            ('mrr', mrr)])

        return log_data
