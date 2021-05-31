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
parser.add_argument('--model_path',
                    type=str,
                    required=True,
                    help='path where the trained model has been saved')

parser.add_argument('--train_file',
                    type=str,
                    required=True,
                    help='path to the training data file')

parser.add_argument('--evaluation_file',
                    type=str,
                    required=True,
                    help='path to the test file')

parser.add_argument('--N_eval',
                    type=int,
                    default=-1,
                    required=False,
                    help='number of points used to training')

args = parser.parse_args()

train_path = args.train_file
eval_path = args.evaluation_file
model_path = args.model_path
N_eval = args.N_eval

set_seeds(0)

# use the training data to get the scaler for the data
R_train, E_train, F_train, _, _, _ = get_train_test_data(path=eval_path, N_train=1000, N_test=0)
energy_scaler = StandardScaler()
_ = energy_scaler.fit(E_train)


# get the data for evaluation
N_eval = N_eval if N_eval > 0 else np.load(eval_path)["E"].shape[0]
_, _, _, R_test, E_test, F_test = get_train_test_data(path=eval_path, N_train=0, N_test=N_eval)
X_dict_test = {'coordinates': R_test}
atom_types = np.load(eval_path)["z"]


# construct the model for the predictions
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

### use the trained model to predict ###
cp_pred = Checkpoint(dirname=model_path)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_pred = FNeuralNet(module=EnergyPredictor,
                      module__geometric_attention_network=GeomAtt,
                      module__atom_types=atom_types,
                      module__with_forces=False,
                      module__F=F2,
                      batch_size=1,
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