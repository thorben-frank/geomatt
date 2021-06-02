import argparse
import matplotlib.pyplot as plt
import numpy as np
from skorch import NeuralNetBinaryClassifier
import torch
from geometric_attention.geometric_classifiers import SchNetClassifier, GeometricAttentionClassifier
from geometric_attention.utils import set_seeds
from pathlib import Path
import os

set_seeds(0)

# Create the parser
parser = argparse.ArgumentParser(description='Reproduce the Geometric Shape Classification.')

# Add the arguments
# Arguments for file and save paths
parser.add_argument('--train_file', type=str, required=True, help='path to the training data file')
parser.add_argument('--figure_folder', type=str, default="", help='path where the figures will be saved')

# Arguments that determine the model
parser.add_argument('--mode', type=str, required=True,
                    help='Pass the MP step to use: Either standard (default) or geometric_attention')

parser.add_argument('--order', type=int, default=2, help='The order k to use when running in mode geometric_attention')
parser.add_argument('--N_epochs', type=int, default=200, help='Number of training epochs')

args = parser.parse_args()

# Read arguments for file and save paths
path = args.train_file
figure_folder = args.figure_folder
order = args.order
mode = args.mode
N_epochs = args.N_epochs

data = np.load(path)
coordinates_train = data["R_train"]
coordinates_test = data["R_test"]
y_train = data["y_train"].astype(np.float32)
y_test = data["y_test"].astype(np.float32)


X_dict_train = {'coordinates': coordinates_train.astype(np.float32)}
X_dict_test = {'coordinates': coordinates_test.astype(np.float32)}

N_atoms = coordinates_train.shape[1]

if mode == "standard":
    model = SchNetClassifier(atom_types=np.ones(N_atoms))
elif mode == "geometric_attention":
    model = GeometricAttentionClassifier(atom_types=np.ones(N_atoms), order=order)
else:
    print("Invalid mode {}".format(mode))

classifier_net = NeuralNetBinaryClassifier(module=model,
                                           optimizer=torch.optim.Adam,
                                           optimizer__lr=1e-4,
                                           max_epochs=N_epochs,
                                           iterator_train__batch_size=4,
                                           iterator_valid__batch_size=4,
                                           iterator_train__shuffle=True,
                                           )

### fit the model ###
classifier_net.fit(X_dict_train, y_train)

test_accuracy = classifier_net.score(X_dict_test, y_test)
print("Achieved Accuracy: {}".format(test_accuracy))

net = classifier_net.module_
net.final_embedding = []
net(torch.tensor(coordinates_test[:100, ...].astype(np.float32)))
embeddings = np.array(net.final_embedding)[0, ...]

c = []
for e in y_test[:100, ...]:
    if e == 0:
        c += ["b"]
    else:
        c += ["r"]

if mode == "standard":
    title = "mode=standard"
    file_name = "mode-standard.pdf"
elif mode == "geometric_attention":
    title = "mode=geometric_attention and k={}".format(order)
    file_name = "mode-geometric-attention-k-{}.pdf".format(order)

fig, ax = plt.subplots(1,1,figsize=(4.5,9))
ax.set_title(title)

ax.scatter(x=np.arange(len(embeddings)), y=embeddings[:], c=c, s=50)
ax.grid()
ax.set_xlabel("test points")
ax.set_ylabel("embedding")
ax.set_xticklabels([])
ax.set_yticklabels([])
#ax.set_ylim(-9, 9)

Path(figure_folder).mkdir(parents=True, exist_ok=True)
file_path = os.path.join(figure_folder, file_name)


plt.savefig(file_path)
