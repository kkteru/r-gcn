import pdb
import torch
from torch import nn

n = 14500
r = 1000
d = 50

i = torch.LongTensor([[0, 1, 1],
                      [2, 0, 2]])
v = torch.FloatTensor([3, 4, 5])
k = torch.sparse.FloatTensor(i, v, torch.Size([n, n])).cuda()

a_list = [k] * r
inp = torch.rand(n, n).cuda()
rel = nn.Parameter(torch.rand(r, n, d)).cuda()

out = torch.zeros(n, d).cuda()
for i, a in enumerate(a_list):
    print(i)
    pdb.set_trace()
    emb_acc = torch.sparse.mm(a, inp)  # (|E| x inp_size)
    out += torch.matmul(emb_acc, rel[i])

