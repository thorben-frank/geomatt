N_eval=${1:--1}

echo
echo Aspirin to Benzene
python eval.py --model_path ../pretrained_models/transfer-learning/aspirin2benzene --train_file ../datasets/MD17/benzene-train.npz --evaluation_file ../datasets/MD17/benzene-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Aspirin to Toluene
python eval.py --model_path ../pretrained_models/transfer-learning/aspirin2toluene --train_file ../datasets/MD17/toluene-train.npz --evaluation_file ../datasets/MD17/toluene-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Benezene to Naphthalene
python eval.py --model_path ../pretrained_models/transfer-learning/benzene2naphthalene --train_file ../datasets/MD17/naphthalene-train.npz --evaluation_file ../datasets/MD17/naphthalene-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Toluene to Benzene
python eval.py --model_path ../pretrained_models/transfer-learning/toluene2benzene --train_file ../datasets/MD17/benzene-train.npz --evaluation_file ../datasets/MD17/benzene-test.npz --N_eval=$N_eval
