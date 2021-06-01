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
# Arguments for file and save paths
parser.add_argument('--model_path', type=str, required=True, help='path where the trained model has been saved')
parser.add_argument('--train_file', type=str, required=True, help='path to the training data file')
parser.add_argument('--evaluation_file', type=str, required=True, help='path to the test file')

# Arguments that determine the model
parser.add_argument('--Nl', type=int, default=3, help='Number of layers per stream. Currently only the same number of layers across streams is allowed.')
parser.add_argument('--Fi', type=int, default=128, help='Inner product dimension')
parser.add_argument('--Fv', type=int, default=128, help='Atom embedding dimension')
parser.add_argument('--orders', type=int, nargs='+', help='The orders k of the streams')
parser.set_defaults(orders=[2, 3, 4])
parser.add_argument('--mode', type=str, help='Set the mode to evaluation or training')
parser.set_defaults(mode="evaluation")

# Arguments that determine the discretization
parser.add_argument('--dmin', type=float, default=0., help='Lower bound of the integral')
parser.add_argument('--dmax', type=float, default=5., help='Upper bound of the integral')
parser.add_argument('--gamma', type=float, default=20., help='RBF width parameter')
parser.add_argument('--interval', type=float, default=.05, help='Spacing between discretization points')

# Arguments that determine the training parameters
parser.add_argument('--batch_size', type=int, default=10, help="Batch size")
parser.add_argument('--forces', dest='forces', action='store_true')
parser.add_argument('--no_forces', dest='forces', action='store_false')
parser.set_defaults(forces=True)
parser.add_argument('--N_eval', type=int, default=-1, required=False, help='Number of points to use for evaluation')

args = parser.parse_args()

# Read the path and file parameters
train_file_path = args.train_file
eval_file_path = args.evaluation_file
model_path = args.model_path

# Read the model hyper parameters
Nl = args.Nl  # No. of streams
Fi = args.Fi  # Inner product dimension
Fv = args.Fv  # Atom embedding dimension
orders = args.orders  # Orders of the streams
mode = args.mode  # Mode of the forward passes

# Read the discretization parameters
dmin = args.dmin
dmax = args.dmax
gamma = args.gamma
interval = args.interval

# Read the evaluation parameters
batch_size = args.batch_size
with_forces = args.forces
N_eval = args.N_eval

set_seeds(0)

# Use the training data to get the scaler for the data
R_train, E_train, F_train, _, _, _ = get_train_test_data(path=train_file_path, N_train=1000, N_test=0)
energy_scaler = StandardScaler()
_ = energy_scaler.fit(E_train)

# Load the data for evaluation
N_eval = N_eval if N_eval > 0 else np.load(eval_file_path)["E"].shape[0]
_, _, _, R_test, E_test, F_test = get_train_test_data(path=eval_file_path, N_train=0, N_test=N_eval)
X_dict_test = {'coordinates': R_test}
atom_types = np.load(eval_file_path)["z"]

# Construct the model for the predictions
streams = [Stream(order=k, F=int(Fi/(2**(k-2))), d_min=dmin, d_max=dmax, interval=interval, gamma=gamma, F_v=Fv, N_L=Nl, mode=mode)
           for k in orders]
GeomAtt = GeometricAttentionNetwork(streams=streams, atom_types=None)

# Use the trained model to predict
cp_pred = Checkpoint(dirname=model_path)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_pred = FNeuralNet(module=EnergyPredictor,
                      module__geometric_attention_network=GeomAtt,
                      module__atom_types=atom_types,
                      module__with_forces=with_forces,
                      module__F=Fv,
                      batch_size=batch_size,
                      criterion=torch.nn.MSELoss,
                      device=dev,
                      )

net_pred.initialize()
net_pred.load_params(checkpoint=cp_pred)
predictions = net_pred.predict(X_dict_test)
energy_pred = energy_scaler.inverse_transform(predictions["E"])
force_pred = energy_scaler.scale_ * predictions["F"]

print("Model Energy MAE = {}".format(np.mean(np.abs(energy_pred.reshape(-1)-E_test.reshape(-1)))))
if len(force_pred) != 0:
    print("Model Force MAE = {}".format(np.mean(np.abs(force_pred.reshape(-1) - F_test.reshape(-1)))))