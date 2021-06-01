# detect-the-interactions-that-matter-in-matter-geometric-attention-for-many-body-systems

## Setup
Clone the code to your local machine using

`git clone https://github.com/thorben-frank/detect-the-interactions-that-matter-in-matter-geometric-attention-for-many-body-systems.git`


All following setup commands are intended to be executed in in the folder in which the 
`setup.py` file lies. 
### Code
To install the package geometric attention as well as
all dependencies which are needed in the following run the command:

`pip install .`
### Getting the Data
In the paper the MD17 data sets and a DNA data set are used. To download the data 
run:

`command to download the data`
### Pretrained Models
To get the pretrained models run:

`command to download pretrained models`

### Sanity Check
You should now have a folder structure which looks the following:

`show the folder tree`

This is important such that the shell scripts which are intended to ease 
reproducability can run properly. If you do not intend to use the shell scripts
and plan to save the pretrained models and the data somewhere else you can still run
the models and evaluate the models using the python scripts in the folder `scripts`. 
These scripts give you the possibility to set all paths per hand.

## Training a Model
Go to the script folder and run the command

`python train.py --help`

which displays all 

 

