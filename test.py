import pdb
import torch

i = torch.LongTensor([[0, 1, 1],
                      [2, 0, 2]])
v = torch.FloatTensor([3, 4, 5])
k = torch.sparse.FloatTensor(i, v, torch.Size([20000, 20000])).cuda()

a_list = [k] * 1000
inp = torch.rand(20000, 20000).cuda()
rel = [torch.rand(20000, 50).cuda()] * 1000

out = torch.zeros(20000, 50).cuda()
for i, a in enumerate(a_list):
    print(i)
    pdb.set_trace()
    emb_acc = torch.sparse.mm(a, inp)  # (|E| x inp_size)
    r = rel[i]
    out += torch.matmul(emb_acc, r)
