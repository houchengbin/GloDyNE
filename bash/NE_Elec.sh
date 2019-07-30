#! /bin/bash
source activate DynNE;

echo '......Start Elec DynWalks bash......';

echo '------------ DynWalks-Elec a02_b05 -----------------------------------------';
echo '------------ DynWalks-Elec a02_b05 -----------------------------------------' > bash/log/Elec_DynWalks_a02_b05.txt;
start=`date +%s`;
python src/main.py --method DynWalks --task save --graph data/Elec/Elec.pkl --emb-file output/Elec_DynWalks.pkl --num-walks 20 --walk-length 80 --window 10 --scheme 3 --limit 0.2 --local-global 0.5 --emb-dim 128 --workers 40  >> bash/log/Elec_DynWalks_a02_b05.txt;
end=`date +%s`;
echo ALL running time: $((end-start)) >> bash/log/Elec_DynWalks_a02_b05.txt;
python src/eval.py --task all --graph data/Elec/Elec.pkl --emb-file output/Elec_DynWalks.pkl >> bash/log/Elec_DynWalks_a02_b05.txt;
echo '--done--' >> bash/log/Elec_DynWalks_a02_b05.txt;
echo '--Elec done--';

echo '......well done......';
