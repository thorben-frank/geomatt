from abc import ABC, abstractmethod

import torch
from torch import nn

from .utils import SSP


class CorrelatorK2(nn.Module, ABC):
    def __init__(self, F, D, interval, shared=False, bias=True):
        super(CorrelatorK2, self).__init__()

        # for k = 2, shared doesn't have any effect and it only accepted for consistency with higher order correlators
        self.shared = shared
        self.interval = interval

        self.Q_0 = nn.Linear(in_features=D, out_features=F, bias=bias)
        self.Q = nn.Linear(in_features=D, out_features=F, bias=bias)

        self.ssp = SSP()

    def forward(self, rbf_0, rbf_d):
        q_0 = self.Q_0(rbf_0)
        # q0.shape = [B,N_v,N_v,F]
        q = self.Q(rbf_d)
        # q.shape = [B,N_v,N_v,F]
        C = (q_0 * q).sum(dim=-1) * self.interval
        # C.shape = [B,N_v,N_v]
        return C


class CorrelatorK3(nn.Module, ABC):
    def __init__(self, F, D, interval, shared=False, bias=True):
        super(CorrelatorK3, self).__init__()

        self.interval = interval
        self.shared = shared
        self.ssp = SSP()

        if not shared:
            self.Q_0 = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.Q = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.R_0 = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.R = nn.Linear(in_features=D, out_features=F, bias=bias)
        else:
            self.Q_0 = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.Q = nn.Linear(in_features=D, out_features=F, bias=bias)

    def forward_not_shared(self, rbf_0, rbf_d):
        q_0 = self.Q_0(rbf_0)
        # q_0.shape = [B,N_v,N_v,F]
        q = self.Q(rbf_d)
        # q.shape = [B,N_v,N_v,F]
        r_0 = self.R_0(rbf_0)
        # r_0.shape = [B,N_v,N_v,F]
        r = self.R(rbf_d)
        # r.shape = [B,N_v,N_v,F]
        C = torch.einsum("bnif,bijf->bnj", q_0 * q, r_0 * r) * self.interval
        # C.shape = [B,N_v,N_v]
        return C

    def forward_shared(self, rbf_0, rbf_d):
        q_0 = self.Q_0(rbf_0)
        # q_0.shape = [B,N_v,N_v,F]
        q = self.Q(rbf_d)
        # q.shape = [B,N_v,N_v,F]
        A_F = q_0 * q
        # A_F.shape = [B,N_v,N_v,F]

        C = torch.einsum("bnif,bijf->bnj", A_F, A_F) * self.interval
        # C.shape = [B,N_v,N_v]
        return C

    def forward(self, rbf_0, rbf_d):
        if self.shared:
            return self.forward_shared(rbf_0=rbf_0, rbf_d=rbf_d)
        else:
            return self.forward_not_shared(rbf_0=rbf_0, rbf_d=rbf_d)


class CorrelatorK4(nn.Module, ABC):
    def __init__(self, F, D, interval, shared=False, bias=True):
        super(CorrelatorK4, self).__init__()

        self.interval = interval
        self.shared = shared
        self.ssp = SSP()

        if not shared:
            self.Q_0 = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.Q = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.R_0 = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.R = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.S_0 = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.S = nn.Linear(in_features=D, out_features=F, bias=bias)
        else:
            self.Q_0 = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.Q = nn.Linear(in_features=D, out_features=F, bias=bias)


    def forward_not_shared(self, rbf_0, rbf_d):
        q_0 = self.Q_0(rbf_0)
        # q_0.shape = [B,N_v,N_v,F]
        q = self.Q(rbf_d)
        # q.shape = [B,N_v,N_v,F]
        r_0 = self.R_0(rbf_0)
        # r_0.shape = [B,N_v,N_v,F]
        r = self.R(rbf_d)
        # r.shape = [B,N_v,N_v,F]
        s_0 = self.S_0(rbf_0)
        # s_0.shape = [B,N_v,N_v,F]
        s = self.S(rbf_d)
        # s.shape = [B,N_v,N_v,F]
        C_ = torch.einsum("bnif,bijf->bnjf", q_0 * q, r_0 * r)
        C = torch.einsum("bnif,bijf->bnj", C_, s_0 * s) * self.interval
        # C.shape = [B,N_v,N_v]
        return C

    def forward_shared(self, rbf_0, rbf_d):
        q_0 = self.Q_0(rbf_0)
        # q_0.shape = [B,N_v,N_v,F]
        q = self.Q(rbf_d)
        # q.shape = [B,N_v,N_v,F]
        A_F = q_0 * q
        C_ = torch.einsum("bnif,bijf->bnjf", A_F, A_F)
        C = torch.einsum("bnif,bijf->bnj", C_, A_F) * self.interval
        # C.shape = [B,N_v,N_v]
        return C

    def forward(self, rbf_0, rbf_d):
        if self.shared:
            return self.forward_shared(rbf_0=rbf_0, rbf_d=rbf_d)
        else:
            return self.forward_not_shared(rbf_0=rbf_0, rbf_d=rbf_d)


class CorrelatorK5(nn.Module, ABC):
    def __init__(self, F, D, interval, shared=False, bias=True):
        super(CorrelatorK5, self).__init__()

        self.interval = interval
        self.shared = shared

        if not shared:
            self.Q_0 = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.Q = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.R_0 = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.R = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.S_0 = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.S = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.T_0 = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.T = nn.Linear(in_features=D, out_features=F, bias=bias)
        else:
            self.Q_0 = nn.Linear(in_features=D, out_features=F, bias=bias)
            self.Q = nn.Linear(in_features=D, out_features=F, bias=bias)

    def forward_not_shared(self, rbf_0, rbf_d):
        q_0 = self.Q_0(rbf_0)
        # q_0.shape = [B,N_v,N_v,F]
        q = self.Q(rbf_d)
        # q.shape = [B,N_v,N_v,F]
        r_0 = self.R_0(rbf_0)
        # r_0.shape = [B,N_v,N_v,F]
        r = self.R(rbf_d)
        # r.shape = [B,N_v,N_v,F]
        s_0 = self.S_0(rbf_0)
        # s_0.shape = [B,N_v,N_v,F]
        s = self.S(rbf_d)
        # s.shape = [B,N_v,N_v,F]
        t_0 = self.T_0(rbf_0)
        # s_0.shape = [B,N_v,N_v,F]
        t = self.T(rbf_d)
        # s.shape = [B,N_v,N_v,F]
        C_ = torch.einsum("bnif,bijf->bnjf", q_0 * q, r_0 * r)
        C_ = torch.einsum("bnif,bijf->bnjf", C_, s_0 * s)
        C = torch.einsum("bnif,bijf->bnj", C_, t_0 * t) * self.interval
        # C.shape = [B,N_v,N_v]
        return C

    def forward_shared(self, rbf_0, rbf_d):
        q_0 = self.Q_0(rbf_0)
        # q_0.shape = [B,N_v,N_v,F]
        q = self.Q(rbf_d)
        # q.shape = [B,N_v,N_v,F]
        A_F = q_0 * q

        C_ = torch.einsum("bnif,bijf->bnjf", A_F, A_F)
        C_ = torch.einsum("bnif,bijf->bnjf", C_, A_F)
        C = torch.einsum("bnif,bijf->bnj", C_, A_F) * self.interval
        # C.shape = [B,N_v,N_v]
        return C

    def forward(self, rbf_0, rbf_d):
        if self.shared:
            return self.forward_shared(rbf_0=rbf_0, rbf_d=rbf_d)
        else:
            return self.forward_not_shared(rbf_0=rbf_0, rbf_d=rbf_d)