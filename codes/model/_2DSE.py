from collections import defaultdict
from typing import List, Any, Tuple, Dict
import math
import torch
import torch.nn as nn
from functools import reduce

class SE(nn.Module):
    def __init__(self):
        super(SE, self).__init__()
        self.g_vol = None
        self.adj: List[Dict[int, float]] = []
        self.node_idx = None
        self.node_deg = None
        self.div_cut = None
        self.div_vol = None
        self.node_in = None
        self.nodes = None
    def __init__v(self, features):

        self.g_vol = None
        self.adj: List[Dict[int, float]] = []
        self.node_idx = None
        self.node_deg = None
        self.div_cut = None
        self.div_vol = None
        self.node_in = None
        self.nodes = None

        self.node_num = features.size(0)
        self.nodes = [i for i in range(self.node_num)]

        fs = [f.flatten() for f in features]
        fs = torch.stack(fs)
        A = torch.corrcoef(fs)
        A.fill_diagonal_(0)
        A[A < 0] = 1e-8
        for i in range(A.size(0)):
            self.adj.append({})
            for j in range(A.size(1)):
                    self.adj[i][j] = A[i, j].item()


        self.g_vol = torch.sum(A).item()
        self.node_deg = torch.sum(A, dim=1).cpu().tolist()
        self.div_vol = torch.sum(A, dim=1).cpu().tolist()
        self.div_cut = torch.sum(A, dim=1).cpu().tolist()
        self.node_in = [0.]*self.node_num
        self.node_div = [i for i in range(self.node_num)]
        self.entropy = [0.]*self.node_num

    def get_clusters(self):
        divs = defaultdict(list)
        for i , div in enumerate(self.node_div):
            divs[div] += [i]

        return list(divs.values())

    def delta_leave(self, i: int, d: int, i_in: float = 0) -> float:
        if math.fabs(self.div_vol[d] - self.node_deg[i]) < 1e-8 and i_in == 0.:
            return 0.

        g_vol = self.g_vol
        i_deg = self.node_deg[i]

        if i_in > 0:
            div_vol = self.div_vol[d] + i_deg
            div_vol_prm = self.div_vol[d]

            div_cut = self.div_cut[d] - 2 * i_in + i_deg
            div_cut_prm = self.div_cut[d]
        else:
            div_vol = self.div_vol[d]
            div_vol_prm = div_vol - i_deg

            div_cut = self.div_cut[d]
            div_cut_prm = div_cut + 2 * self.node_in[i] - i_deg

        try:
            div_prm_se = - (div_cut_prm / g_vol) * math.log(div_vol_prm / g_vol)
        except ValueError as e:
            print(e)
            div_prm_se = 0.

        try:
            div_se = (div_cut / g_vol) * math.log(div_vol / g_vol)
        except ValueError as e:
            print(e)
            div_se = 0.
        try:
            str_se = (div_vol_prm / g_vol) * math.log(div_vol_prm / div_vol)
        except ValueError as e:
            print(e)
            str_se = 0.

        i_se = (i_deg / g_vol) * math.log(g_vol / div_vol)

        return div_prm_se + div_se + str_se + i_se

    def strategy(self, i: int):
        src_d = self.node_div[i]
        tgt_d = src_d

        delta_l = self.delta_leave(i, src_d)
        self.entropy[i] = delta_l

        if delta_l < 0:
            tgt_d = -1

        adj_div = {}
        for j, w in self.adj[i].items():
            div_j = self.node_div[j]
            if div_j != src_d:
                adj_div[div_j] = adj_div.get(div_j, 0) + w

        i_in = 0
        delta_min = 0
        for k, ii in adj_div.items():
            delta_lk = self.delta_leave(i, k, ii)
            delta_tk = delta_l - delta_lk

            if delta_tk < delta_min:
                delta_min = delta_tk
                tgt_d = k
                i_in = ii

        return tgt_d, i_in, delta_min

    def _leave(self, i):
        # update clust C_k
        d = self.node_div[i]
        self.div_vol[d] -= self.node_deg[i]
        self.div_cut[d] = self.div_cut[d] + 2 * self.node_in[i] - self.node_deg[i]

        # new cluster {x}
        self.node_div[i] = i
        self.div_vol[i] = self.node_deg[i]
        self.div_cut[i] = self.div_vol[i]
        self.node_in[i] = 0

    def _transfer(self, i, tgt_d, i_in):
        # update clust C_src
        src_d = self.node_div[i]
        self.div_vol[src_d] -= self.node_deg[i]
        if self.div_vol[src_d] > 0:
            self.div_cut[src_d] = self.div_cut[src_d] + 2 * self.node_in[i] - self.node_deg[i]
        else:
            self.div_cut[src_d] = 0

        # update clust C_src
        self.node_div[i] = tgt_d
        self.div_vol[tgt_d] += self.node_deg[i]
        self.div_cut[tgt_d] = self.div_cut[tgt_d] - 2 * i_in + self.node_deg[i]
        self.node_in[i] = i_in

        for j, w in self.adj[i].items():
            if self.node_div[j] == src_d:
                self.node_in[j] -= w
            if self.node_div[j] == tgt_d:
                self.node_in[j] += w
    def forward(self, x, max_iter=10):
        self.__init__v(x)
        for it in range(max_iter):

            delta_sum = 0
            leave, transfer = 0, 0

            for i in self.nodes:
                tgt_div, cut_i, se_delta = self.strategy(i)

                if se_delta < 0:
                    if tgt_div < 0:
                        # strategy 1
                        self._leave(i)
                        leave += 1
                    else:
                        # strategy 2
                        self._transfer(i, tgt_div, cut_i)
                        transfer += 1

                    delta_sum += se_delta

            if leave + transfer == 0:
                break

        divisions = self.get_clusters()

        choosen_f = []
        choosen_se = []
        for c in divisions:
            if len(c) > 1:
                best = max(c, key=lambda i: self.entropy[i])
                choosen_f.append(best)
                choosen_se.append(self.entropy[best])
            else:
                choosen_f.append(c[0])
                choosen_se.append(self.entropy[c[0]])
        choosen_se = torch.tensor(choosen_se, dtype=torch.float32, device=x.device)
        choosen_weight = torch.softmax(choosen_se, dim=0).view(-1, 1, 1, 1)

        return choosen_f, choosen_weight




