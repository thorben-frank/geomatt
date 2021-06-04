import numpy as np

import torch
from torch.autograd.functional import jacobian
from torch import nn
from .utils import EnergyExtractor, composite_function, adjacency


class EnergyPredictor(nn.Module):
    def __init__(self, geometric_attention_network, F, atom_types, linear=False, with_forces=False):
        # atom_types.shape = [N_v]
        super(EnergyPredictor, self).__init__()

        self.F = F

        self.C_embeddings = nn.Parameter(torch.empty(size=(1, 1, int(self.F))), requires_grad=True)
        nn.init.xavier_uniform_(self.C_embeddings.data, gain=1.414)
        self.H_embeddings = nn.Parameter(torch.empty(size=(1, 1, int(self.F))), requires_grad=True)
        nn.init.xavier_uniform_(self.H_embeddings.data, gain=1.414)
        self.Li_embeddings = nn.Parameter(torch.empty(size=(1, 1, int(self.F))), requires_grad=True)
        nn.init.xavier_uniform_(self.Li_embeddings.data, gain=1.414)
        self.O_embeddings = nn.Parameter(torch.empty(size=(1, 1, int(self.F))), requires_grad=True)
        nn.init.xavier_uniform_(self.O_embeddings.data, gain=1.414)

        self.atom_embedding_dict = {1: self.H_embeddings, 3: self.Li_embeddings, 6: self.C_embeddings,
                                    8: self.O_embeddings}

        if 7 in atom_types:
            self.N_embeddings = nn.Parameter(torch.empty(size=(1, 1, int(self.F))), requires_grad=True)
            nn.init.xavier_uniform_(self.N_embeddings.data, gain=1.414)
            self.atom_embedding_dict[7] = self.N_embeddings

        if isinstance(atom_types, torch.Tensor):
            self.atom_types = atom_types.tolist()
        elif isinstance(atom_types, np.ndarray):
            self.atom_types = torch.tensor(atom_types.astype(np.float32)).tolist()
        else:
            print("Unsupported atom type type")

        self.geom_att = geometric_attention_network
        self.N_s = len(self.geom_att.streams)

        #self.W = nn.Linear(in_features=self.F, out_features=self.F)

        self.energy_extractor = EnergyExtractor(F=int(self.N_s * self.F), F_=int(self.N_s * self.F / 2), linear=linear)

        self.with_forces = with_forces

        self.attentions = []

        #self.final_embeddings = []

    def freeze_attention_module(self):
        for param in self.geom_att.parameters():
            param.requires_grad = False

    def forward_energy(self, coordinates):
        # coordinates.shape = [B,N_v,3]
        B = coordinates.shape[0]
        N_v = coordinates.shape[1]

        atom_embeddings = torch.cat([self.atom_embedding_dict[i] for i in self.atom_types], dim=-2)
        # atom_embeddings.shape = [1,N_v,F]
        atom_embeddings = atom_embeddings.repeat(B, 1, 1)
        # atom_embeddings.shape = [B,N_v,F]

        v_L = self.geom_att(coordinates=coordinates, atom_embeddings=atom_embeddings)
        # v_L.shape = [B,N_s,N_v,F]

        v_L = v_L.permute(0, 2, 1, 3).reshape(B, N_v, -1)
        # v_L.shape = [B,N_v,N_s x F]
        E = self.energy_extractor(v_L).sum(dim=-2)
        # E.shape = [B,1]
        return E

    def forward_forces(self, coordinates):
        forces = - jacobian(composite_function(torch.sum, self.forward_energy), coordinates, create_graph=True).squeeze(1)
        # squeeze removes the extra dimension in jac that comes from the fact that the output is of shape [1]
        # forces.shape = [B,N_v,3] where D is the dimension of the input space

        return forces

    def forward(self, coordinates):
        energy = self.forward_energy(coordinates)
        if self.with_forces:
            forces = self.forward_forces(coordinates)
        else:
            forces = torch.tensor([])

        return energy, forces


class GeometricAttentionNetwork(nn.Module):
    def __init__(self, streams, atom_types=None, training=True):
        super(GeometricAttentionNetwork, self).__init__()

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(streams, nn.ModuleList):
            self.streams = streams
        else:
            self.streams = nn.ModuleList()
            for s in streams:
                self.streams.append(s)

        # currently only same discretization for all streams is supported
        d_min = self.streams[-1].d_min
        d_max = self.streams[-1].d_max
        interval = self.streams[-1].interval
        self.D = int(abs(d_max - d_min) / interval)

        self.gamma = self.streams[-1].gamma
        self.mu_k = torch.arange(d_min, d_max, interval).to(dev)
        self.mu_k.requires_grad = False

        self.atom_functions = None

        if atom_types is not None:
            print("Providing atom types to GeomAtt is currently not supported but comes soon ;)")
            #self.atom_types = []

            #if isinstance(atom_types, torch.Tensor):
            #    self.atom_types = atom_types.tolist()
            #elif isinstance(atom_types, np.ndarray):
            #    self.atom_types = torch.tensor(atom_types.astype(np.float32)).tolist()
            #else:
            #    print("Unsupported atom type type")

            #if 1 in self.atom_types:
            #    self.f_H = nn.Parameter(torch.empty(size=(1, 1, 1, int(self.D))), requires_grad=True)
            #    nn.init.xavier_uniform_(self.f_H.data, gain=1.414)
            #if 6 in self.atom_types:
            #    self.f_C = nn.Parameter(torch.empty(size=(1, 1, 1, int(self.D))), requires_grad=True)
            #    nn.init.xavier_uniform_(self.f_C.data, gain=1.414)
            #if 8 in self.atom_types:
            #    self.f_O = nn.Parameter(torch.empty(size=(1, 1, 1, int(self.D))), requires_grad=True)
            #    nn.init.xavier_uniform_(self.f_O.data, gain=1.414)

            #self.atom_functions = {1: self.f_H, 6: self.f_C, 8: self.f_O}

    def rbf_expansion(self, adj):
        # adj.shape = [B,N_v,N_v,1]
        B = adj.shape[0]
        N_v = adj.shape[1]
        D = self.mu_k.shape[-1]

        adj_rbf = torch.exp(
            -self.gamma * (adj.repeat(1, 1, 1, D) - self.mu_k.view(1, 1, 1, -1).repeat(B, N_v, N_v, 1)) ** 2)
        # adj_rbf.shape = [B,N_v,N_v,D]
        return adj_rbf

    def forward(self, coordinates, atom_embeddings):
        # coordinates.shape = [B,N_v,3]
        B = coordinates.shape[0]
        N_v = coordinates.shape[1]


        #adj_atoms_base = torch.cat([self.atom_functions[i] for i in self.atom_types], dim=-3)
        # adj_atoms_base.shape = [1,N_v,1,D]

        #adj_atoms_start = adj_atoms_base.repeat(B, 1, N_v, 1)
        # adj_atoms_start.shape = [B,N_v,N_v,D]

        #adj_atoms_target = adj_atoms_start.permute(0, 2, 1, 3)
        # adj_atoms_target.shape = [B,N_v,N_v,D]

        adj = adjacency(coordinates=coordinates).unsqueeze(-1)
        # adj.shape = [B,N_v,N_v,1]

        rbf_0 = self.rbf_expansion(torch.zeros_like(adj))
        # rbf0.shape = [B,N_v,N_v,D]

        rbf_d = self.rbf_expansion(adj)
        # rbf_d.shape = [B,N_v,N_v,D]

        v_L = []
        for stream in self.streams:
            v_L += [stream(rbf_0=rbf_0, rbf_d=rbf_d, atom_embeddings=atom_embeddings, N_s=len(self.streams))]

        # len(v_L) = N_s w/ N_s no. of streams
        # v_L[entry].shape = [B,N_v,F]
        v_L = torch.stack(v_L, dim=1)
        # v_L.shape = [B,N_s,N_v,F]

        return v_L
