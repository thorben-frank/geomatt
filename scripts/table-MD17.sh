N_eval=${1:--1}

echo

echo Aspirin:
python eval.py --model_path ../pretrained_models/aspirin --train_file ../datasets/aspirin-train.npz --evaluation_file ../datasets/aspirin-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Benzene:
python eval.py --model_path ../pretrained_models/benzene --train_file ../datasets/benzene-train.npz --evaluation_file ../datasets/benzene-test.npz --N_eval=$N_eval 
echo -------------------------------------------------
echo Ethanol:
python eval.py --model_path ../pretrained_models/ethanol --train_file ../datasets/ethanol-train.npz --evaluation_file ../datasets/ethanol-test.npz --N_eval=$N_eval 
echo -------------------------------------------------
echo Malonaldehyde:
python eval.py --model_path ../pretrained_models/malonaldehyde --train_file ../datasets/malonaldehyde-train.npz --evaluation_file ../datasets/malonaldehyde-test.npz --N_eval=$N_eval 
echo -------------------------------------------------
echo Naphthalene:
python eval.py --model_path ../pretrained_models/naphthalene --train_file ../datasets/naphthalene-train.npz --evaluation_file ../datasets/naphthalene-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Salicyclic Acid:
python eval.py --model_path ../pretrained_models/salicyclic --train_file ../datasets/salicyclic-train.npz --evaluation_file ../datasets/salicyclic-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Toluene:
python eval.py --model_path ../pretrained_models/toluene --train_file ../datasets/toluene-train.npz --evaluation_file ../datasets/toluene-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Uracil:
python eval.py --model_path ../pretrained_models/uracil --train_file ../datasets/uracil-train.npz --evaluation_file ../datasets/uracil-test.npz --N_eval=$N_eval
