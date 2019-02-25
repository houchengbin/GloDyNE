#! /bin/bash
echo '......Chengbin HOU @SUSTech 2019......' > bash/log/LFR_DynRWSG_prob.txt;
echo 'LFR aane lp with diff link sparsity......' >> bash/log/LFR_DynRWSG_prob.txt;
echo '......good luck......' >> bash/log/LFR_DynRWSG_prob.txt;

echo '-------------------restart-prob 0.0 -----------------------------------------' >> bash/log/LFR_DynRWSG_prob.txt;
echo '-------------------restart-prob 0.0 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/LFR/LFR.pkl --emb-file output/LFR_DynRWSG_prob_128_embs.pkl --num-walks 20 --restart-prob 0.0 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_DynRWSG_prob.txt;
echo '--done--';
echo '--done--' >> bash/log/LFR_DynRWSG_prob.txt;

echo '-------------------restart-prob 0.1 -----------------------------------------' >> bash/log/LFR_DynRWSG_prob.txt;
echo '-------------------restart-prob 0.1 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/LFR/LFR.pkl --emb-file output/LFR_DynRWSG_prob_128_embs.pkl --num-walks 20 --restart-prob 0.1 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_DynRWSG_prob.txt;
echo '--done--';
echo '--done--' >> bash/log/LFR_DynRWSG_prob.txt;

echo '-------------------restart-prob 0.2 -----------------------------------------' >> bash/log/LFR_DynRWSG_prob.txt;
echo '-------------------restart-prob 0.2 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/LFR/LFR.pkl --emb-file output/LFR_DynRWSG_prob_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_DynRWSG_prob.txt;
echo '--done--';
echo '--done--' >> bash/log/LFR_DynRWSG_prob.txt;

echo '-------------------restart-prob 0.3 -----------------------------------------' >> bash/log/LFR_DynRWSG_prob.txt;
echo '-------------------restart-prob 0.3 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/LFR/LFR.pkl --emb-file output/LFR_DynRWSG_prob_128_embs.pkl --num-walks 20 --restart-prob 0.3 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_DynRWSG_prob.txt;
echo '--done--';
echo '--done--' >> bash/log/LFR_DynRWSG_prob.txt;

echo '-------------------restart-prob 0.4 -----------------------------------------' >> bash/log/LFR_DynRWSG_prob.txt;
echo '-------------------restart-prob 0.4 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/LFR/LFR.pkl --emb-file output/LFR_DynRWSG_prob_128_embs.pkl --num-walks 20 --restart-prob 0.4 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_DynRWSG_prob.txt;
echo '--done--';
echo '--done--' >> bash/log/LFR_DynRWSG_prob.txt;

echo '-------------------restart-prob 0.5 -----------------------------------------' >> bash/log/LFR_DynRWSG_prob.txt;
echo '-------------------restart-prob 0.5 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/LFR/LFR.pkl --emb-file output/LFR_DynRWSG_prob_128_embs.pkl --num-walks 20 --restart-prob 0.5 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_DynRWSG_prob.txt;
echo '--done--';
echo '--done--' >> bash/log/LFR_DynRWSG_prob.txt;

echo '-------------------restart-prob 0.6 -----------------------------------------' >> bash/log/LFR_DynRWSG_prob.txt;
echo '-------------------restart-prob 0.6 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/LFR/LFR.pkl --emb-file output/LFR_DynRWSG_prob_128_embs.pkl --num-walks 20 --restart-prob 0.6 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_DynRWSG_prob.txt;
echo '--done--';
echo '--done--' >> bash/log/LFR_DynRWSG_prob.txt;

echo '-------------------restart-prob 0.7 -----------------------------------------' >> bash/log/LFR_DynRWSG_prob.txt;
echo '-------------------restart-prob 0.7 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/LFR/LFR.pkl --emb-file output/LFR_DynRWSG_prob_128_embs.pkl --num-walks 20 --restart-prob 0.7 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_DynRWSG_prob.txt;
echo '--done--';
echo '--done--' >> bash/log/LFR_DynRWSG_prob.txt;

echo '-------------------restart-prob 0.8 -----------------------------------------' >> bash/log/LFR_DynRWSG_prob.txt;
echo '-------------------restart-prob 0.8 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/LFR/LFR.pkl --emb-file output/LFR_DynRWSG_prob_128_embs.pkl --num-walks 20 --restart-prob 0.8 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_DynRWSG_prob.txt;
echo '--done--';
echo '--done--' >> bash/log/LFR_DynRWSG_prob.txt;

echo '-------------------restart-prob 0.9 -----------------------------------------' >> bash/log/LFR_DynRWSG_prob.txt;
echo '-------------------restart-prob 0.9 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/LFR/LFR.pkl --emb-file output/LFR_DynRWSG_prob_128_embs.pkl --num-walks 20 --restart-prob 0.9 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_DynRWSG_prob.txt;
echo '--done--';
echo '--done--' >> bash/log/LFR_DynRWSG_prob.txt;

echo '-------------------restart-prob 1.0 -----------------------------------------' >> bash/log/LFR_DynRWSG_prob.txt;
echo '-------------------restart-prob 1.0 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/LFR/LFR.pkl --emb-file output/LFR_DynRWSG_prob_128_embs.pkl --num-walks 20 --restart-prob 1.0 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_DynRWSG_prob.txt;
echo '--done--';
echo '--done--' >> bash/log/LFR_DynRWSG_prob.txt;

echo '......well done......' >> bash/log/LFR_DynRWSG_prob.txt;