N_eval=${1:--1}

echo
echo Aspirin to Benzene
python eval.py --model_path ../pretrained_models/aspirin2benzene --train_file ../datasets/benzene-train.npz --evaluation_file ../datasets/benzene-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Aspirin to Toluene
python eval.py --model_path ../pretrained_models/aspirin2toluene --train_file ../datasets/toluene-train.npz --evaluation_file ../datasets/toluene-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Benezene to Naphthalene
python eval.py --model_path ../pretrained_models/benzene2naphthalene --train_file ../datasets/naphthalene-train.npz --evaluation_file ../datasets/naphthalene-test.npz --N_eval=$N_eval
echo -------------------------------------------------
echo Toluene to Benzene
python eval.py --model_path ../pretrained_models/toluene2benzene --train_file ../datasets/benzene-train.npz --evaluation_file ../datasets/benzene-test.npz --N_eval=$N_eval
