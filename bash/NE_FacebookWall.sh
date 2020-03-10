#! /bin/bash
source activate GloDyNE

echo '......Chengbin HOU @SUSTech 2019 FacebookWall 0.1 DynWalks  bash......';

echo '------------ FacebookWall_DynWalks_S4.txt-----------------------------------------';
echo '------------ FacebookWall_DynWalks_S4.txt-----------------------------------------' > bash/log/FacebookWall_DynWalks_S4.txt;
start=`date +%s`;
python src/main.py --method DynWalks --task save --graph data/FacebookWall/FacebookWall_new.pkl --label data/FacebookWall/FacebookWall_label.pkl --emb-file output/FacebookWall_DynWalks.pkl --num-walks 10 --walk-length 80 --window 10 --limit 0.1 --scheme 4 --seed 2019 --emb-dim 128 --workers 32  >> bash/log/FacebookWall_DynWalks_S4.txt;
end=`date +%s`;
echo ALL running time: $((end-start)) >> bash/log/FacebookWall_DynWalks_S4.txt;
python src/eval.py --task all --graph data/FacebookWall/FacebookWall_new.pkl --emb-file output/FacebookWall_DynWalks.pkl --label data/FacebookWall/FacebookWall_label.pkl --seed 2019 >> bash/log/FacebookWall_DynWalks_S4.txt;
echo '--done--' >> bash/log/FacebookWall_DynWalks_S4.txt;

echo '......well done......'