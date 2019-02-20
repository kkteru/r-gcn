from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pdb


class Evaluator():
    def __init__(self, params, encoder, decoder, link_data_sampler, neg_sample_size=0):
        self.encoder = encoder
        self.decoder = decoder

        self.link_data_sampler = link_data_sampler
        # self.neg_sample_size = neg_sample_size if neg_sample_size != 0 else len(link_data_sampler.data)
        self.params = params

    # Find the rank of ground truth head in the distance array,
    # If (head, num, rel) in all_data,
    # skip without counting.
    def _filter(self, head, tail, rel, array, rank, h):
        filtered_rank = rank
        for i in range(rank):
            if (head * (1 - h) + array[i] * h, array[i] * (1 - h) + tail * h, rel) in self.link_data_sampler.all_data:
                filtered_rank = filtered_rank - 1
        return filtered_rank

    def get_log_data(self, eval_mode='head'):
        # pdb.set_trace()

        mr = []
        mrr = []
        hit10 = []

        if eval_mode == 'head' or eval_mode == 'avg':

            distArrayHead = self.decoder.get_all_scores(self.encoder.final_emb.data, self.link_data_sampler.data[:, 0],
                                                        self.link_data_sampler.data[:, 1], self.link_data_sampler.data[:, 2],
                                                        'head').cpu().numpy()
            rankArrayHead = np.argsort(distArrayHead, axis=1)

            # Don't check whether it is false negative
            rankListHead = [int(np.argwhere(elem[1] == elem[0])) for elem in zip(self.link_data_sampler.data[:, 0], rankArrayHead)]
            if self.params.filter:
                rankListHead = [int(self._filter(elem[0], elem[1], elem[2], elem[3], elem[4], h=1))
                                for elem in zip(self.link_data_sampler.data[:, 0], self.link_data_sampler.data[:, 1],
                                                self.link_data_sampler.data[:, 2], rankArrayHead, rankListHead)]

            isHit10ListHead = [x for x in rankListHead if x < 10]

            assert len(rankListHead) == len(self.link_data_sampler.data)

            mr.append(np.mean(rankListHead))
            mrr.append(np.mean(1 / np.array(rankListHead)))
            hit10.append(len(isHit10ListHead) / len(rankListHead))

# -------------------------------------------------------------------- #

        if eval_mode == 'tail' or eval_mode == 'avg':
            distArrayTail = self.decoder.get_all_scores(self.encoder.final_emb.data, self.link_data_sampler.data[:, 0],
                                                        self.link_data_sampler.data[:, 1], self.link_data_sampler.data[:, 2],
                                                        'tail').cpu().numpy()
            rankArrayTail = np.argsort(distArrayTail, axis=1)

            # Don't check whether it is false negative
            rankListTail = [int(np.argwhere(elem[1] == elem[0])) for elem in zip(self.link_data_sampler.data[:, 1], rankArrayTail)]
            if self.params.filter:
                rankListTail = [int(self._filter(elem[0], elem[1], elem[2], elem[3], elem[4], h=0))
                                for elem in zip(self.link_data_sampler.data[:, 0], self.link_data_sampler.data[:, 1],
                                                self.link_data_sampler.data[:, 2], rankArrayTail, rankListTail)]

            isHit10ListTail = [x for x in rankListTail if x < 10]

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
