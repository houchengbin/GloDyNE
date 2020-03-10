#! /bin/bash
source activate GloDyNE

echo '......Chengbin HOU @SUSTech 2019 Cora 0.1 DynWalks  bash......';

echo '------------ Cora_DynWalks_S4.txt-----------------------------------------';
echo '------------ Cora_DynWalks_S4.txt-----------------------------------------' > bash/log/Cora_DynWalks_S4.txt;
start=`date +%s`;
python src/main.py --method DynWalks --task save --graph data/Cora/Cora_new.pkl --label data/Cora/Cora_label.pkl --emb-file output/Cora_DynWalks.pkl --num-walks 10 --walk-length 80 --window 10 --limit 0.1 --scheme 4 --seed 2019 --emb-dim 128 --workers 32  >> bash/log/Cora_DynWalks_S4.txt;
end=`date +%s`;
echo ALL running time: $((end-start)) >> bash/log/Cora_DynWalks_S4.txt;
python src/eval.py --task all --graph data/Cora/Cora_new.pkl --emb-file output/Cora_DynWalks.pkl --label data/Cora/Cora_label.pkl --seed 2019 >> bash/log/Cora_DynWalks_S4.txt;
echo '--done--' >> bash/log/Cora_DynWalks_S4.txt;

echo '......well done......'