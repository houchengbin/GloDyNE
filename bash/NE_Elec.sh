#! /bin/bash
source activate GloDyNE

echo '......Chengbin HOU @SUSTech 2019 Elec 0.1 DynWalks  bash......';

echo '------------ Elec_DynWalks_S4.txt-----------------------------------------';
echo '------------ Elec_DynWalks_S4.txt-----------------------------------------' > bash/log/Elec_DynWalks_S4.txt;
start=`date +%s`;
python src/main.py --method DynWalks --task save --graph data/Elec/Elec_new.pkl --label data/Elec/Elec_label.pkl --emb-file output/Elec_DynWalks.pkl --num-walks 10 --walk-length 80 --window 10 --limit 0.1 --scheme 4 --seed 2019 --emb-dim 128 --workers 32  >> bash/log/Elec_DynWalks_S4.txt;
end=`date +%s`;
echo ALL running time: $((end-start)) >> bash/log/Elec_DynWalks_S4.txt;
python src/eval.py --task all --graph data/Elec/Elec_new.pkl --emb-file output/Elec_DynWalks.pkl --label data/Elec/Elec_label.pkl --seed 2019 >> bash/log/Elec_DynWalks_S4.txt;
echo '--done--' >> bash/log/Elec_DynWalks_S4.txt;

echo '......well done......'