import argparse
from geomatt.stream import Stream
from geomatt.network import GeometricAttentionNetwork, EnergyPredictor
from geomatt.utils import set_seeds, get_train_test_data, read_json_file
from geomatt.skorch_extensions import FNeuralNet
import json
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from skorch.callbacks import Checkpoint
import torch

# Create the parser
parser = argparse.ArgumentParser(description='Train GeomAtt on a molecule of your choice.')

# Add the arguments
# Arguments for file and save paths
parser.add_argument('--train_file', type=str, required=True, help='path to the training data file')
parser.add_argument('--pretrained_path', type=str, required=True, help='path where the pretrained model is saved')
parser.add_argument('--save_path', type=str, required=True, help='path where the trained model will be saved')

# Arguments that determine the training parameters
parser.add_argument('--N_epochs', type=int, default=1000, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=10, help="Batch size")
parser.add_argument('--rho', type=float, default=.01, help="Trade off parameter between energy and force loss")
parser.add_argument('--forces', dest='forces', action='store_true', help="Training with forces (Default)")
parser.add_argument('--no_forces', dest='forces', action='store_false', help="Training without forces")
parser.set_defaults(forces=True)

args = parser.parse_args()

# Read arguments for file and save paths
train_file_path = args.train_file
checkpoint_path = args.save_path
pretrained_path = args.pretrained_path

pretrained_hyperparameter_path = os.path.join(pretrained_path, "hyperparameters.json")
h = read_json_file(pretrained_hyperparameter_path)

# Read the model hyper parameters
Nl = h["Nl"]  # No. of streams
Fi = h["Fi"]  # Inner product dimension
Fv = h["Fv"]  # Atom embedding dimension
orders = h["orders"]  # Orders of the streams

# Read the discretization parameters
dmin = h["dmin"]
dmax = h["dmax"]
gamma = h["gamma"]
interval = h["interval"]

# Read the training parameters
N_epochs = args.N_epochs
h["N_epochs"] = N_epochs
batch_size = args.batch_size
h["batch_size"] = batch_size
with_forces = args.forces
h["forces"] = with_forces
rho = args.rho if args.forces else 1.
h["rho"] = rho

Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
file_path = os.path.join(checkpoint_path, "hyperparameters.json")
with open(file_path, 'w') as f:
    json.dump(h, f)

set_seeds(0)

# Read the data of the transfer molecule
R_train, E_train, F_train, _, _, _ = get_train_test_data(path=train_file_path, N_train=1000, N_test=0)

energy_scaler = StandardScaler()
energies_train = energy_scaler.fit_transform(E_train).astype(np.float32)
forces_train = F_train / energy_scaler.scale_
atom_types = np.load(train_file_path)["z"]

X_dict_train = {'coordinates': R_train}
Y_dict_train = {'E': energies_train.astype(np.float32), 'F': forces_train.astype(np.float32)}

# Construct the model
streams = [Stream(order=k, F=int(Fi/(2**(k-2))), d_min=dmin, d_max=dmax, interval=interval, gamma=gamma, F_v=Fv, N_L=Nl, mode="training")
           for k in orders]

GeomAtt = GeometricAttentionNetwork(streams=streams, atom_types=None)

# saves the model parameters which gave smallest validation error
cp = Checkpoint(dirname=checkpoint_path)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_net = FNeuralNet(module=EnergyPredictor,
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
                            callbacks=[cp],
                            device=dev,
                            )

pretrained_net.load_params(checkpoint=Checkpoint(dirname=pretrained_path))

# Freeze all parameters except for the last layers
for param in pretrained_net.module_.parameters():
    param.requires_grad = False
for param in pretrained_net.module_.energy_extractor.parameters():
    param.requires_grad = True

# Fit the model
pretrained_net.partial_fit(X_dict_train, Y_dict_train)


