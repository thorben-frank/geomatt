
echo SHAPES A
echo --------------------------------------------------------------------------------
echo Standard Step 
python3 geometric-classifiers.py --train_file ../datasets/shapes-A.npz --figure_folder ../shapes/shapes-A --mode standard --N_epochs=100
echo -------------------------------------------------
echo Geometric Attention k=2
python3 geometric-classifiers.py --train_file ../datasets/shapes-A.npz --figure_folder ../shapes/shapes-A --mode geometric_attention --N_epochs=100 --order 2
echo -------------------------------------------------
echo Geometric Attention k=3
python3 geometric-classifiers.py --train_file ../datasets/shapes-A.npz --figure_folder ../shapes/shapes-A --mode geometric_attention --N_epochs=100 --order 3
echo --------------------------------------------------------------------------------
echo
echo
echo SHAPES B
echo --------------------------------------------------------------------------------
echo Geometric Attention k=3
python3 geometric-classifiers.py --train_file ../datasets/shapes-B.npz --figure_folder ../shapes/shapes-B --mode geometric_attention --N_epochs=100 --order 3
echo -------------------------------------------------
echo Geometric Attention k=4
python3 geometric-classifiers.py --train_file ../datasets/shapes-B.npz --figure_folder ../shapes/shapes-B --mode geometric_attention --N_epochs=100 --order 4
echo --------------------------------------------------------------------------------
