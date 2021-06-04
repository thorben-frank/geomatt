import argparse
from geomatt.stream import Stream
from geomatt.network import GeometricAttentionNetwork, EnergyPredictor
from geomatt.utils import set_seeds, get_train_test_data
from geomatt.skorch_extensions import FNeuralNet
import json
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from skorch.callbacks import Checkpoint
from skorch.callbacks import LRScheduler

import torch

# Create the parser
parser = argparse.ArgumentParser(description='Train GeomAtt on a molecule of your choice.')

# Add the arguments
# Arguments for file and save paths
parser.add_argument('--train_file', type=str, required=True, help='path to the training data file')
parser.add_argument('--save_path', type=str, required=True, help='path where the trained model will be saved')

# Arguments that determine the model
parser.add_argument('--Nl', type=int, default=3, help='Number of layers per stream. Currently only the same number of layers across streams is allowed.')
parser.add_argument('--Fi', type=int, default=128, help='Inner product dimension')
parser.add_argument('--Fv', type=int, default=128, help='Atom embedding dimension')
parser.add_argument('--orders', type=int, nargs='+', help='The orders k of the streams')
parser.set_defaults(orders=[2, 3, 4])

# Arguments that determine the discretization
parser.add_argument('--dmin', type=float, default=0., help='Lower bound of the integral')
parser.add_argument('--dmax', type=float, default=5., help='Upper bound of the integral')
parser.add_argument('--gamma', type=float, default=20., help='RBF width parameter')
parser.add_argument('--interval', type=float, default=.05, help='Spacing between discretization points')

# Arguments that determine the training parameters
parser.add_argument('--N_epochs', type=int, default=2000, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=10, help="Batch size")
parser.add_argument('--rho', type=float, default=.01, help="Trade off parameter between energy and force loss")
parser.add_argument('--forces', dest='forces', action='store_true', help="Training with forces (Default)")
parser.add_argument('--no_forces', dest='forces', action='store_false', help="Training without forces")
parser.set_defaults(forces=True)

args = parser.parse_args()

# Read arguments for file and save paths
path = args.train_file
checkpoint_path = args.save_path

# Create hyperparameters file
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
file_path = os.path.join(checkpoint_path, "hyperparameters.json")
h = args.__dict__
h.pop('train_file', None)
h.pop('save_path', None)
with open(file_path, 'w') as f:
    json.dump(h, f)

# Read the model hyper parameters
Nl = args.Nl  # No. of streams
Fi = args.Fi  # Inner product dimension
Fv = args.Fv  # Atom embedding dimension
orders = args.orders  # Orders of the streams

# Read the discretization parameters
dmin = args.dmin
dmax = args.dmax
gamma = args.gamma
interval = args.interval

# Read the training parameters
N_epochs = args.N_epochs
batch_size = args.batch_size
with_forces = args.forces
rho = args.rho if args.forces else 1.

set_seeds(0)

# Read the data
R_train, E_train, F_train, _, _, _ = get_train_test_data(path=path, N_train=1000, N_test=0)

energy_scaler = StandardScaler()
energies_train = energy_scaler.fit_transform(E_train).astype(np.float32)
forces_train = F_train / energy_scaler.scale_
atom_types = np.load(path)["z"]

X_dict_train = {'coordinates': R_train}
Y_dict_train = {'E': energies_train.astype(np.float32), 'F': forces_train.astype(np.float32)}

# Construct the model
streams = [Stream(order=k, F=int(Fi/(2**(k-2))), d_min=dmin, d_max=dmax, interval=interval, gamma=gamma, F_v=Fv, N_L=Nl, mode="training")
           for k in orders]

GeomAtt = GeometricAttentionNetwork(streams=streams, atom_types=None)

# saves the model parameters which gave smallest validation error
cp = Checkpoint(dirname=checkpoint_path)

lrs = LRScheduler(policy="StepLR", step_every="epoch", step_size=1000, gamma=.96, event_name="step_lr")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = FNeuralNet(module=EnergyPredictor,
                 module__geometric_attention_network=GeomAtt,
                 module__atom_types=atom_types,
                 module__with_forces=with_forces,
                 module__F=Fv,
                 optimizer=torch.optim.Adam,
                 optimizer__lr=1e-4,
                 criterion=torch.nn.MSELoss,
                 max_epochs=N_epochs,
                 beta=rho,
                 iterator_train__batch_size=batch_size,
                 iterator_valid__batch_size=batch_size,
                 iterator_train__shuffle=True,
                 callbacks=[cp, lrs],
                 device=dev,
                 )

# Fit the model
net.fit(X_dict_train, Y_dict_train)