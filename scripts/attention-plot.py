import argparse
from geometric_attention.stream import Stream
from geometric_attention.network import GeometricAttentionNetwork, EnergyPredictor
from geometric_attention.utils import set_seeds, get_train_test_data, read_json_file
from geometric_attention.skorch_extensions import FNeuralNet
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
from skorch.callbacks import Checkpoint
import torch


# Create the parser
parser = argparse.ArgumentParser(description='Train GeomAtt on a molecule of your choice.')

# Add the arguments
# Arguments for file and save paths
parser.add_argument('--model_path', type=str, required=True, help='path where the trained model has been saved')
parser.add_argument('--evaluation_file', type=str, required=True, help='path to the test file')
parser.add_argument('--figure_folder', type=str, required=False, help='folder where the plot will be saved')

# Arguments for the plot
parser.add_argument('--i', type=str, default=1, help='the i-th evaluation point is used for the plots (default=1)')
parser.add_argument('--L', type=str, default=1, help='what layer to use (default=1)')
parser.add_argument('--k', type=str, default=2, help='what correlation order to use (default=2)')

# Arguments for Evaluation
parser.add_argument('--N_eval', type=int, default=10, required=False, help='Number of points to use for evaluation')

args = parser.parse_args()

# Read the path and file parameters
eval_file_path = args.evaluation_file
model_path = args.model_path
figure_folder = "." if not args.figure_folder else args.figure_folder

# Read the evaluation parameters
N_eval = args.N_eval

# Read the plot parameters
layer = int(args.L)-1
index = int(args.i)-1
order = int(args.k)-2

# Read the model hyperparameters
hyperparameter_path = os.path.join(model_path, "hyperparameters.json")
h = read_json_file(hyperparameter_path)

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

set_seeds(0)

# Load the data for evaluation
_, _, _, R_test, E_test, F_test = get_train_test_data(path=eval_file_path, N_train=0, N_test=N_eval)
X_dict_test = {'coordinates': R_test}
atom_types = np.load(eval_file_path)["z"]

# Construct the model for the predictions
streams = [Stream(order=k, F=int(Fi/(2**(k-2))), d_min=dmin, d_max=dmax, interval=interval, gamma=gamma, F_v=Fv, N_L=Nl, mode="evaluation")
           for k in orders]
GeomAtt = GeometricAttentionNetwork(streams=streams, atom_types=None)

# Use the trained model to predict
cp_pred = Checkpoint(dirname=model_path)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_pred = FNeuralNet(module=EnergyPredictor,
                      module__geometric_attention_network=GeomAtt,
                      module__atom_types=atom_types,
                      module__with_forces=False,
                      module__F=Fv,
                      batch_size=1,
                      criterion=torch.nn.MSELoss,
                      device=dev,
                      )

net_pred.initialize()
net_pred.load_params(checkpoint=cp_pred)
_ = net_pred.predict(X_dict_test)


def get_atom_label(x):
    if x == 1:
        return "H"
    elif x == 6:
        return "C"
    elif x == 7:
        return "N"
    elif x == 8:
        return "O"


batch = 0
atts = np.array(net_pred.module_.geom_att.streams[order].attentions)[index, layer, batch, ...]
atts = (abs(atts).T + abs(atts))/2
atts = atts - np.diag(np.diagonal(atts))
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
cb = ax.imshow(atts[:29, :29], cmap="Greys")
# as we are only looking at a single GC in the GC-GC pair the resulting attention matrix is cut out
# as all other molecules have less than 29 atoms, this is fine
truncated_atom_types = atom_types[:29]
labels = [get_atom_label(x) for x in truncated_atom_types]
ax.set_xticks(np.arange(len(truncated_atom_types)))
ax.set_xticklabels(labels)
ax.set_yticks(np.arange(len(truncated_atom_types)))
ax.set_yticklabels(labels)

Path(figure_folder).mkdir(parents=True, exist_ok=True)

figure_path = os.path.join(figure_folder, "attention-matrix-k{}-L{}-i{}.pdf".format(args.k, args.L, args.i))
plt.savefig(figure_path)



