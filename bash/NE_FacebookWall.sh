#! /bin/bash
source activate DynNE;

echo '......Start FacebookWall DynWalks bash......';

echo '------------ DynWalks-FacebookWall a02_b05 -----------------------------------------';
echo '------------ DynWalks-FacebookWall a02_b05 -----------------------------------------' > bash/log/FacebookWall_DynWalks_a02_b05.txt;
start=`date +%s`;
python src/main.py --method DynWalks --task save --graph data/FacebookWall/FacebookWall.pkl --emb-file output/FacebookWall_DynWalks.pkl --num-walks 20 --walk-length 80 --window 10 --scheme 3 --limit 0.2 --local-global 0.5 --emb-dim 128 --workers 40  >> bash/log/FacebookWall_DynWalks_a02_b05.txt;
end=`date +%s`;
echo ALL running time: $((end-start)) >> bash/log/FacebookWall_DynWalks_a02_b05.txt;
python src/eval.py --task all --graph data/FacebookWall/FacebookWall.pkl --emb-file output/FacebookWall_DynWalks.pkl >> bash/log/FacebookWall_DynWalks_a02_b05.txt;
echo '--done--' >> bash/log/FacebookWall_DynWalks_a02_b05.txt;
echo '--FacebookWall done--';

echo '......well done......';
