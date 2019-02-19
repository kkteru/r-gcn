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
            if (head * (1 - h) + array[i] * h, array[i] * (1 - h) + tail * h, rel) in self.data_sampler.all_data:
                filtered_rank = filtered_rank - 1
        return filtered_rank

    def get_log_data(self, eval_mode='head'):
        # pdb.set_trace()

        h_e = self.encoder.final_emb.data.cpu().numpy()[self.link_data_sampler.data[:, 0]]
        t_e = self.encoder.final_emb.data.cpu().numpy()[self.link_data_sampler.data[:, 1]]
        r_e = self.encoder.final_emb.data.cpu().numpy()[self.link_data_sampler.data[:, 2]]

        mr = []
        hit10 = []

        if eval_mode == 'head' or eval_mode == 'avg':
            c_h_e = t_e - r_e

            distHead = pairwise_distances(c_h_e, self.encoder.final_emb.weight.data.cpu().numpy(), metric='manhattan')

            rankArrayHead = np.argsort(distHead, axis=1)

            # Don't check whether it is false negative
            rankListHead = [int(np.argwhere(elem[1] == elem[0])) for elem in zip(self.data_sampler.data[:, 0], rankArrayHead)]
            if self.params.filter:
                rankListHead = [int(self._filter(elem[0], elem[1], elem[2], elem[3], elem[4], h=1))
                                for elem in zip(self.data_sampler.data[:, 0], self.data_sampler.data[:, 1],
                                                self.data_sampler.data[:, 2], rankArrayHead, rankListHead)]

            isHit10ListHead = [x for x in rankListHead if x < 10]

            assert len(rankListHead) == len(self.data_sampler.data)

            mr.append(np.mean(rankListHead))
            hit10.append(len(isHit10ListHead) / len(rankListHead))

# -------------------------------------------------------------------- #

        if eval_mode == 'tail' or eval_mode == 'avg':
            c_t_e = h_e + r_e

            distTail = pairwise_distances(c_t_e, self.model.ent_embeddings.weight.data.cpu().numpy(), metric='manhattan')

            rankArrayTail = np.argsort(distTail, axis=1)

            # Don't check whether it is false negative
            rankListTail = [int(np.argwhere(elem[1] == elem[0])) for elem in zip(self.data_sampler.data[:, 1], rankArrayTail)]
            if self.params.filter:
                rankListTail = [int(self._filter(elem[0], elem[1], elem[2], elem[3], elem[4], h=0))
                                for elem in zip(self.data_sampler.data[:, 0], self.data_sampler.data[:, 1],
                                                self.data_sampler.data[:, 2], rankArrayTail, rankListTail)]

            isHit10ListTail = [x for x in rankListTail if x < 10]

            assert len(rankListTail) == len(self.data_sampler.data)

            mr.append(np.mean(rankListTail))
            hit10.append(len(isHit10ListTail) / len(rankListTail))

        mr = np.mean(mr)
        hit10 = np.mean(hit10)

        log_data = dict([
            ('hit@10', hit10),
            ('mr', mr)])

        return log_data
