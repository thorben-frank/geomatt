python attention-plot.py --model_path ../pretrained_models/DNA/CG-CG/ --evaluation_file ../datasets/DNA/cg-cg-test.npz --figure_folder ../figures/figure_6b --L=1 --k=2 --i=1 --N_eval=1
python attention-plot.py --model_path ../pretrained_models/DNA/CG-CG/ --evaluation_file ../datasets/DNA/cg-cg-test.npz --figure_folder ../figures/figure_6c --L=2 --k=2 --i=1 --N_eval=1
python attention-plot.py --model_path ../pretrained_models/MD17/aspirin/ --evaluation_file ../datasets/MD17/aspirin-test.npz --figure_folder ../figures/figure5_upper --L=1 --k=2 --i=1 --N_eval=1
python attention-plot.py --model_path ../pretrained_models/transfer-learning/aspirin2benzene/ --evaluation_file ../datasets/MD17/benzene-test.npz --figure_folder ../figures/figure5_lower --L=1 --k=2 --i=1 --N_eval=1


