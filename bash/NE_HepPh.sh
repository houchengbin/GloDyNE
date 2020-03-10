#! /bin/bash
source activate GloDyNE

echo '......Chengbin HOU @SUSTech 2019 HepPh 0.1 DynWalks  bash......';

echo '------------ HepPh_DynWalks_S4.txt-----------------------------------------';
echo '------------ HepPh_DynWalks_S4.txt-----------------------------------------' > bash/log/HepPh_DynWalks_S4.txt;
start=`date +%s`;
python src/main.py --method DynWalks --task save --graph data/HepPh/HepPh_new.pkl --label data/HepPh/HepPh_label.pkl --emb-file output/HepPh_DynWalks.pkl --num-walks 10 --walk-length 80 --window 10 --limit 0.1 --scheme 4 --seed 2019 --emb-dim 128 --workers 32  >> bash/log/HepPh_DynWalks_S4.txt;
end=`date +%s`;
echo ALL running time: $((end-start)) >> bash/log/HepPh_DynWalks_S4.txt;
python src/eval.py --task all --graph data/HepPh/HepPh_new.pkl --emb-file output/HepPh_DynWalks.pkl --label data/HepPh/HepPh_label.pkl --seed 2019 >> bash/log/HepPh_DynWalks_S4.txt;
echo '--done--' >> bash/log/HepPh_DynWalks_S4.txt;

echo '......well done......'