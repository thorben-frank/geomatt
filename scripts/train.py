import argparse
from geometric_attention.stream import Stream
from geometric_attention.network import GeometricAttentionNetwork, EnergyPredictor
from geometric_attention.utils import set_seeds, get_train_test_data
from geometric_attention.skorch_extensions import FNeuralNet
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from skorch.callbacks import Checkpoint
import torch

# Create the parser
parser = argparse.ArgumentParser(description='Train GeomAtt on a molecule of your choice.')

# Add the arguments
parser.add_argument('--input_file',
                    type=str,
                    required=True,
                    help='path to the training data file')

parser.add_argument('--save_path',
                    type=str,
                    help='path where the trained model is saved')

args = parser.parse_args()

path = args.input_file
checkpoint_path = args.save_path if args.save_path else os.path.join("..", "user_trained_model")

set_seeds(0)
R_train, E_train, F_train, _, _, _ = get_train_test_data(path=path, N_train=1000, N_test=0)

energy_scaler = StandardScaler()
energies_train = energy_scaler.fit_transform(E_train).astype(np.float32)
forces_train = F_train / energy_scaler.scale_
atom_types = np.load(path)["z"]


orders = [2, 3, 4]
stream_F = 64
d_min = 0
d_max = 5
interval = .1
gamma = 10.
shared = True
bias = True

F2 = 64

streams = [Stream(order=k, F=stream_F, d_min=d_min, d_max=d_max, interval=interval, gamma=gamma, shared=shared, bias=bias, F_v=F2, N_L=2, mode="test")
           for k in orders]

GeomAtt = GeometricAttentionNetwork(streams=streams, atom_types=None)

# saves the model parameters which gave smallest validation error
cp = Checkpoint(dirname=checkpoint_path)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = FNeuralNet(module=EnergyPredictor,
                 module__geometric_attention_network=GeomAtt,
                 module__atom_types=atom_types,
                 module__with_forces=False,
                 module__F=F2,
                 module__linear=False,
                 optimizer=torch.optim.Adam,
                 optimizer__lr=1e-4,
                 criterion=torch.nn.MSELoss,
                 max_epochs=75,
                 beta=1.,
                 iterator_train__batch_size=4,
                 iterator_valid__batch_size=4,
                 iterator_train__shuffle=True,
                 callbacks=[cp],
                 device=dev,
                 )

X_dict_train = {'coordinates': R_train}
Y_dict_train = {'E': energies_train.astype(np.float32), 'F': forces_train.astype(np.float32)}

### fit the model ###
net.fit(X_dict_train, Y_dict_train)