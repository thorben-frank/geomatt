import torch
from torch import nn
from .utils import SSP
from .correlator import CorrelatorK2, CorrelatorK3, CorrelatorK4, CorrelatorK5


def request_correlator(k, F, D, interval, shared=False, bias=True):
    if k == 2:
        return CorrelatorK2(F=F, D=D, interval=interval, shared=shared, bias=bias)
    elif k == 3:
        return CorrelatorK3(F=F, D=D, interval=interval, shared=shared, bias=bias)
    elif k == 4:
        return CorrelatorK4(F=F, D=D, interval=interval, shared=shared, bias=bias)
    elif k == 5:
        return CorrelatorK5(F=F, D=D, interval=interval, shared=shared, bias=bias)
    else:
        raise (NotImplementedError("Order k = {} is not implemented".format(k)))


class Stream(nn.Module):
    def __init__(self, order, N_L=1, F=16, F_v=16, d_min=0., d_max=5., interval=.1, gamma=.5, shared=False, bias=True, correlators=None, mode="training"):
        super(Stream, self).__init__()

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.k = order

        self.F = F
        self.D = int(abs(d_max - d_min) / interval)

        self.d_min = d_min
        self.d_max = d_max
        self.interval = interval

        self.gamma = gamma

        self.shared = shared
        self.bias = bias

        self.ssp = SSP()

        self.N_L = N_L

        if correlators is None:
            self.interactions = nn.ModuleList(
                [Interaction(request_correlator(k=order, F=self.F, D=self.D, interval=self.interval, shared=shared, bias=bias), F_vertex=F_v) for n in range(self.N_L)])
        else:
            self.interactions = nn.ModuleList([Interaction(c, F_vertex=F_v) for c in correlators])
        self.attentions = []
        self.mode = mode

    def forward(self, rbf_0, rbf_d, atom_embeddings, N_s):
        # rbf_0.shape = [B,N_v,N_v,D]
        # rbf_d.shape = [B,N_v,N_v,D]
        # atom_embeddings.shape = [B,N_v,F]
        v_l = [atom_embeddings]

        atts = []
        for inter in self.interactions:
            alpha, v = inter(rbf_0=rbf_0, rbf_d=rbf_d, atom_embeddings=v_l[-1], N_s=N_s)
            if self.mode != "training":
                atts += [alpha.cpu().detach().numpy()]
            v_l += [v]

        if self.mode != "training":
            self.attentions += [atts]
        # len(v_l) = N_s
        # v_l[entry].shape = [B,N_v,F]

        return v_l[-1]


class Interaction(nn.Module):
    def __init__(self, correlator, F_vertex):
        super(Interaction, self).__init__()

        self.correlator = correlator
        self.W = nn.Linear(in_features=F_vertex, out_features=F_vertex)

    def forward(self, rbf_0, rbf_d, atom_embeddings, N_s):
        # rbf_0.shape = [B,N_v,N_v,D]
        # rbf_d.shape = [B,N_v,N_v,D]
        # atom_embeddings.shape = [B,N_v,F]

        x = self.W(atom_embeddings)
        # x.shape = [B,N_v,F]

        alpha = 1/N_s * self.correlator(rbf_0=rbf_0, rbf_d=rbf_d)
        # alpha.shape = [B,N_v,N_v]

        x = torch.einsum("bnm,bmf->bnf", alpha, x)
        # x.shape = [B,N_v,F]

        v_l = atom_embeddings + x
        # v_l.shape = [B,N_v,F]

        return alpha, v_l
