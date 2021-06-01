N_eval=${1:--1}

echo

echo Aspirin:
python eval.py --model_path ../pretrained_models/MD17/aspirin --train_file ../datasets/MD17/aspirin-train.npz --evaluation_file ../datasets/MD17/aspirin-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Benzene:
python eval.py --model_path ../pretrained_models/MD17/benzene --train_file ../datasets/MD17/benzene-train.npz --evaluation_file ../datasets/MD17/benzene-test.npz --N_eval=$N_eval 
echo -------------------------------------------------
echo Ethanol:
python eval.py --model_path ../pretrained_models/MD17/ethanol --train_file ../datasets/MD17/ethanol-train.npz --evaluation_file ../datasets/MD17/ethanol-test.npz --N_eval=$N_eval 
echo -------------------------------------------------
echo Malonaldehyde:
python eval.py --model_path ../pretrained_models/MD17/malonaldehyde --train_file ../datasets/MD17/malonaldehyde-train.npz --evaluation_file ../datasets/MD17/malonaldehyde-test.npz --N_eval=$N_eval 
echo -------------------------------------------------
echo Naphthalene:
python eval.py --model_path ../pretrained_models/MD17/naphthalene --train_file ../datasets/MD17/naphthalene-train.npz --evaluation_file ../datasets/MD17/naphthalene-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Salicyclic Acid:
python eval.py --model_path ../pretrained_models/MD17/salicyclic --train_file ../datasets/MD17/salicyclic-train.npz --evaluation_file ../datasets/MD17/salicyclic-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Toluene:
python eval.py --model_path ../pretrained_models/MD17/toluene --train_file ../datasets/MD17/toluene-train.npz --evaluation_file ../datasets/MD17/toluene-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Uracil:
python eval.py --model_path ../pretrained_models/MD17/uracil --train_file ../datasets/MD17/uracil-train.npz --evaluation_file ../datasets/MD17/uracil-test.npz --N_eval=$N_eval
