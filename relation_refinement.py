from typing import Tuple, List, Dict
import torch
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class Refinement(torch.nn.Module):
    def __init__(
            self,sizes: Tuple[int, int], edim: int, rdim: int, gatecell: str,
            init_size: float = 1e-3,
    ):
        super(Refinement, self).__init__()
        self.sizes = sizes
        self.edim = edim
        self.rdim = rdim
        self.gatecell = gatecell
        self.lhs = torch.nn.Embedding( sizes[0], edim).cuda()
        self.rel = torch.nn.Embedding(sizes[1], rdim).cuda()
        self.rhs = torch.nn.Embedding( sizes[0], edim).cuda()


        self.gate = {
            'RNNCell': lambda: torch.nn.RNNCell(rdim, edim),
            'LSTMCell': lambda: torch.nn.LSTMCell(rdim, edim),
            'GRUCell': lambda: torch.nn.GRUCell(rdim, edim)
        }[gatecell]().cuda()

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size


    def forward(self, lhs_idx,rel_idx):#x.shape:batch_size*3
        lhs = self.lhs(lhs_idx)
        rel = self.rel(rel_idx)

        if self.gatecell == 'LSTMCell':
            c = torch.zeros_like(lhs)
            rel_update, c1 = self.gate(rel, (lhs, c)) 
        else:
            rel_update = self.gate(rel, lhs)
        return rel_update

class N3(torch.nn.Module):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight  # 0.01

    def forward(self, factors):#factor:tuple(lsh,rel,rhs)
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]

