#! /bin/bash
source activate DynNE;

echo '......Start AS733 DynWalks bash......';

echo '------------ DynWalks-AS733 a02_b05 -----------------------------------------';
echo '------------ DynWalks-AS733 a02_b05 -----------------------------------------' > bash/log/AS733_DynWalks_a02_b05.txt;
start=`date +%s`;
python src/main.py --method DynWalks --task save --graph data/AS733/AS733.pkl --emb-file output/AS733_DynWalks.pkl --num-walks 20 --walk-length 80 --window 10 --scheme 3 --limit 0.2 --local-global 0.5 --emb-dim 128 --workers 40  >> bash/log/AS733_DynWalks_a02_b05.txt;
end=`date +%s`;
echo ALL running time: $((end-start)) >> bash/log/AS733_DynWalks_a02_b05.txt;
python src/eval.py --task all --graph data/AS733/AS733.pkl --emb-file output/AS733_DynWalks.pkl >> bash/log/AS733_DynWalks_a02_b05.txt;
echo '--done--' >> bash/log/AS733_DynWalks_a02_b05.txt;
echo '--AS733 done--';

echo '......well done......';
