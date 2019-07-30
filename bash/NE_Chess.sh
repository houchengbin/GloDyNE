#! /bin/bash
source activate DynNE;

echo '......Start Chess DynWalks bash......';

echo '------------ DynWalks-Chess a02_b05 -----------------------------------------';
echo '------------ DynWalks-Chess a02_b05 -----------------------------------------' > bash/log/Chess_DynWalks_a02_b05.txt;
start=`date +%s`;
python src/main.py --method DynWalks --task save --graph data/Chess/Chess.pkl --emb-file output/Chess_DynWalks.pkl --num-walks 20 --walk-length 80 --window 10 --scheme 3 --limit 0.2 --local-global 0.5 --emb-dim 128 --workers 40  >> bash/log/Chess_DynWalks_a02_b05.txt;
end=`date +%s`;
echo ALL running time: $((end-start)) >> bash/log/Chess_DynWalks_a02_b05.txt;
python src/eval.py --task all --graph data/Chess/Chess.pkl --emb-file output/Chess_DynWalks.pkl >> bash/log/Chess_DynWalks_a02_b05.txt;
echo '--done--' >> bash/log/Chess_DynWalks_a02_b05.txt;
echo '--Chess done--';

echo '......well done......';
