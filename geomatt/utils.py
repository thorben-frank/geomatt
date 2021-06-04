import errno
import itertools as it
import json
import numpy as np
import os
import scipy.linalg as LA

import torch
from torch import nn
import torch.nn.functional as _F


class EnergyExtractor(nn.Module):
    def __init__(self, F, F_, linear=False):
        super(EnergyExtractor, self).__init__()

        self.Q = nn.Linear(in_features=int(F), out_features=F_)
        self.R = nn.Linear(in_features=F_, out_features=1)

        if linear is False:
            self.activation_function = SSP()
        else:
            self.activation_function = nn.Identity()

    def forward(self, x):
        return self.R(self.activation_function(self.Q(x)))


class SSP(nn.Softplus):
    r"""Applies the element-wise function:
    .. math::
        \text{SSP}(x) = \text{Softplus}(x) - \text{Softplus}(0)
    The default SSP looks like ELU(alpha=log2).
    Args:
        beta: the :math:`\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> m = SSP()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__(beta, threshold)
        self.sp0 = _F.softplus(torch.zeros(1), self.beta, self.threshold).item()

    def forward(self, input):
        return _F.softplus(input, self.beta, self.threshold) - self.sp0


def composite_function(f, g):
    return lambda x : f(g(x))


def set_seeds(seed=0):
    #print("setting random seed to: ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    try:
        random.seed(seed)
    except NameError:
        import random
        random.seed(seed)

    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_train_test_data(path, N_train, N_test, seed=0):
    np.random.seed(seed)
    data = np.load(path)
    p = np.random.permutation(data["R"].shape[0])

    E = data["E"][p, ...]
    R = data["R"][p, ...]
    F = data["F"][p, ...]
    Z = data["z"]

    E_train = E[:N_train, ...]
    R_train = R[:N_train, ...]
    F_train = F[:N_train, ...]

    E_test = E[N_train:N_train + N_test, ...]
    R_test = R[N_train:N_train + N_test, ...]
    F_test = F[N_train:N_train + N_test, ...]

    return R_train.astype(np.float32), E_train.astype(np.float32), F_train.astype(np.float32), R_test.astype(
        np.float32), E_test.astype(np.float32), F_test.astype(np.float32)


def vectors_to_distances(x):
    return np.array([[LA.norm(x[k,i,:]-x[k,j,:]) for i,j in it.combinations(np.arange(x.shape[-2]),2)]
                     for k in np.arange(x.shape[0])])


def adjacency(coordinates):
    # coordinates has shape (B,N_v,3)

    # number of batches
    B = coordinates.shape[0]
    # number of vertices
    N_v = coordinates.shape[1]

    coords_repeated_in_chunks = coordinates.repeat_interleave(N_v, dim=1)
    coords_repeated_alternating = coordinates.repeat(1, N_v, 1)
    pair_distances = torch.norm(coords_repeated_in_chunks - coords_repeated_alternating, dim=2)
    adj = torch.where(pair_distances != 0, pair_distances, torch.zeros_like(pair_distances))
    adj = adj.view(B, N_v, N_v)
    return adj


def read_json_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    data = None
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
