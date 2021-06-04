import numpy as np
import torch
from torch import nn
from .utils import adjacency, SSP
from .stream import Stream


class Interaction(nn.Module):
    def __init__(self, conv):
        super(Interaction, self).__init__()

        self.conv = conv

        F = self.conv.F
        self.W = nn.Linear(in_features=F, out_features=F)
        self.Q = nn.Linear(in_features=F, out_features=F)
        self.R = nn.Linear(in_features=F, out_features=F)

        self.ssp = SSP()

    def forward(self, coordinates, atom_emb):
        # coordinates.shape = [B,N_v,3]
        # atom_emb.shape = [B,N_v,F]
        B = coordinates.shape[0]
        N_v = coordinates.shape[-2]
        F = coordinates.shape[-1]

        x = self.W(atom_emb)
        # x.shape = [B,N_v,F]

        W = self.conv(coordinates)
        # K.shape = [B,N_v,N_v,F]

        W = W * (torch.ones(1, N_v, N_v, 1) - torch.eye(N_v).view(1, N_v, N_v, 1))
        x = torch.einsum("bnmf,bmf->bnf", W, x)
        # x.shape = [B,N_v,F]

        # x =  self.R(self.ssp(self.Q(x)))
        # x.shape = [B,N_v,F]

        atom_emb_ = atom_emb + x
        # atom_emb_.shape = [B,N_v,F]

        return atom_emb_


class CfConv(nn.Module):
    def __init__(self, F=16, d_min=0, d_max=2, interval=0.1, gamma=10):
        super(CfConv, self).__init__()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.F = F

        self.d_min = d_min
        self.d_max = d_max
        self.interval = interval
        self.gamma = gamma
        self.D = int(abs(d_max - d_min) / interval)

        self.mu_k = torch.arange(d_min, d_max, interval).to(dev)
        self.mu_k.requires_grad = False

        self.W = nn.Linear(in_features=self.D, out_features=F)
        self.Q = nn.Linear(in_features=F, out_features=F)
        self.R = nn.Linear(in_features=F, out_features=F)

        self.ssp = SSP()

    def RBF(self, adj):
        # adj.shape = [B,N_v,N_v,1]
        B = adj.shape[0]
        N_v = adj.shape[1]

        adj_rbf = torch.exp(
            -self.gamma * (adj.repeat(1, 1, 1, self.D) - self.mu_k.view(1, 1, 1, -1).repeat(B, N_v, N_v, 1)) ** 2)

        return adj_rbf

    def forward(self, coordinates):
        # coordinates.shape = [B,N_v,3]
        adj = adjacency(coordinates).unsqueeze(-1)
        # adj.shape = [B,N_v,N_v,1]
        adj_rbf = self.RBF(adj)
        # adj_rbf.shape = [B,N_v,N_v,D]
        x = self.W(adj_rbf)
        # x.shape = [B,N_v,N_v,F]
        return x


class SchNetClassifier(nn.Module):
    def __init__(self, atom_types, order=None):
        # atom_types.shape = [N_v]
        super(SchNetClassifier, self).__init__()

        self.cfconv = CfConv()
        self.interaction = Interaction(conv=self.cfconv)
        self.F = self.cfconv.F

        if isinstance(atom_types, torch.Tensor):
            self.atom_types = atom_types.tolist()
        elif isinstance(atom_types, np.ndarray):
            self.atom_types = torch.tensor(atom_types.astype(np.float32)).tolist()
        else:
            print("Unsupported atom type type")

        self.atom_embeddings = nn.Parameter(torch.empty(size=(1, 1, int(self.F))), requires_grad=True)
        nn.init.xavier_uniform_(self.atom_embeddings.data, gain=1.414)

        self.atom_embedding_dict = {1: self.atom_embeddings}

        self.Q = nn.Linear(in_features=int(self.F), out_features=1)

        self.ssp = SSP()

        self.final_embedding = []

    def forward(self, coordinates):
        # coordinates.shape = [B,N_v,3]
        B = coordinates.shape[0]

        atom_embeddings = torch.cat([self.atom_embedding_dict[i] for i in self.atom_types], dim=1).repeat(B, 1, 1)
        # atom_embeddings.shape = [1,N_v,F]

        v_ = self.interaction(coordinates=coordinates, atom_emb=atom_embeddings)[:, 0, :]
        # v_.shape = [B,F]
        v_ = self.Q(v_)

        self.final_embedding += [v_.detach().cpu().numpy()]

        return v_


class GeometricAttentionClassifier(nn.Module):
    def __init__(self, atom_types, order):
        super(GeometricAttentionClassifier, self).__init__()

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(atom_types, torch.Tensor):
            self.atom_types = atom_types.tolist()
        elif isinstance(atom_types, np.ndarray):
            self.atom_types = torch.tensor(atom_types.astype(np.float32)).tolist()
        else:
            print("Unsupported atom type type")

        # currently only same discretization for all streams is supported
        d_min = 0
        d_max = 2
        interval = 0.1
        self.D = int(abs(d_max - d_min) / interval)

        self.gamma = 10
        self.mu_k = torch.arange(d_min, d_max, interval).to(dev)
        self.mu_k.requires_grad = False

        self.stream = Stream(order=order, N_L=1, F=16, F_v=1, d_min=d_min, d_max=d_max, interval=interval)
        self.F_v = 1

        self.atom_embeddings = nn.Parameter(torch.empty(size=(1, 1, int(self.F_v))), requires_grad=True)
        nn.init.xavier_uniform_(self.atom_embeddings.data, gain=1.414)

        self.atom_embedding_dict = {1: self.atom_embeddings}

        self.Q = nn.Linear(in_features=int(self.F_v), out_features=1)

        self.final_embedding = []

    def rbf_expansion(self, adj):
        # adj.shape = [B,N_v,N_v,1]
        B = adj.shape[0]
        N_v = adj.shape[1]
        D = self.mu_k.shape[-1]

        adj_rbf = torch.exp(
            -self.gamma * (adj.repeat(1, 1, 1, D) - self.mu_k.view(1, 1, 1, -1).repeat(B, N_v, N_v, 1)) ** 2)
        # adj_rbf.shape = [B,N_v,N_v,D]
        return adj_rbf

    def forward(self, coordinates):
        # coordinates.shape = [B,N_v,3]
        B = coordinates.shape[0]
        N_v = coordinates.shape[1]

        adj = adjacency(coordinates=coordinates).unsqueeze(-1)
        # adj.shape = [B,N_v,N_v,1]

        rbf_0 = self.rbf_expansion(torch.zeros_like(adj))
        # rbf0.shape = [B,N_v,N_v,D]

        rbf_d = self.rbf_expansion(adj)
        # rbf_d.shape = [B,N_v,N_v,D]

        atom_embeddings = torch.cat([self.atom_embedding_dict[i] for i in self.atom_types], dim=1).repeat(B, 1, 1)

        v_ = self.stream(rbf_0=rbf_0, rbf_d=rbf_d, atom_embeddings=atom_embeddings, N_s=1)[:, 0, :]
        # v_.shape = [B,F]
        v_ = self.Q(v_)
        self.final_embedding += [v_.detach().cpu().numpy()]

        return v_


