#! /bin/bash
source activate GloDyNE

echo '......Chengbin HOU @SUSTech 2019 DBLP 0.1 DynWalks  bash......';

echo '------------ DBLP_DynWalks_S4.txt-----------------------------------------';
echo '------------ DBLP_DynWalks_S4.txt-----------------------------------------' > bash/log/DBLP_DynWalks_S4.txt;
start=`date +%s`;
python src/main.py --method DynWalks --task save --graph data/DBLP/DBLP_new.pkl --label data/DBLP/DBLP_label.pkl --emb-file output/DBLP_DynWalks.pkl --num-walks 10 --walk-length 80 --window 10 --limit 0.1 --scheme 4 --seed 2019 --emb-dim 128 --workers 32  >> bash/log/DBLP_DynWalks_S4.txt;
end=`date +%s`;
echo ALL running time: $((end-start)) >> bash/log/DBLP_DynWalks_S4.txt;
python src/eval.py --task all --graph data/DBLP/DBLP_new.pkl --emb-file output/DBLP_DynWalks.pkl --label data/DBLP/DBLP_label.pkl --seed 2019 >> bash/log/DBLP_DynWalks_S4.txt;
echo '--done--' >> bash/log/DBLP_DynWalks_S4.txt;

echo '......well done......'