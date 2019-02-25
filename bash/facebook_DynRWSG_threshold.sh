#! /bin/bash
echo '......Chengbin HOU @SUSTech 2019......' > bash/log/facebook_DynRWSG_threshold.txt;
echo 'facebook aane lp with diff link sparsity......' >> bash/log/facebook_DynRWSG_threshold.txt;
echo '......good luck......' >> bash/log/facebook_DynRWSG_threshold.txt;

echo '-------------------update-threshold 0.0 -----------------------------------------' >> bash/log/facebook_DynRWSG_threshold.txt;
echo '-------------------update-threshold 0.0 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/facebook/facebook.pkl --emb-file output/facebook_DynRWSG_threshold_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.0 --emb-dim 128 --workers 40  >> bash/log/facebook_DynRWSG_threshold.txt;
echo '--done--';
echo '--done--' >> bash/log/facebook_DynRWSG_threshold.txt;

echo '-------------------update-threshold 0.05 -----------------------------------------' >> bash/log/facebook_DynRWSG_threshold.txt;
echo '-------------------update-threshold 0.05 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/facebook/facebook.pkl --emb-file output/facebook_DynRWSG_threshold_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.05 --emb-dim 128 --workers 40  >> bash/log/facebook_DynRWSG_threshold.txt;
echo '--done--';
echo '--done--' >> bash/log/facebook_DynRWSG_threshold.txt;

echo '-------------------update-threshold 0.1 -----------------------------------------' >> bash/log/facebook_DynRWSG_threshold.txt;
echo '-------------------update-threshold 0.1 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/facebook/facebook.pkl --emb-file output/facebook_DynRWSG_threshold_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/facebook_DynRWSG_threshold.txt;
echo '--done--';
echo '--done--' >> bash/log/facebook_DynRWSG_threshold.txt;

echo '-------------------update-threshold 0.15 -----------------------------------------' >> bash/log/facebook_DynRWSG_threshold.txt;
echo '-------------------update-threshold 0.15 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/facebook/facebook.pkl --emb-file output/facebook_DynRWSG_threshold_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.15 --emb-dim 128 --workers 40  >> bash/log/facebook_DynRWSG_threshold.txt;
echo '--done--';
echo '--done--' >> bash/log/facebook_DynRWSG_threshold.txt;

echo '-------------------update-threshold 0.2 -----------------------------------------' >> bash/log/facebook_DynRWSG_threshold.txt;
echo '-------------------update-threshold 0.2 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/facebook/facebook.pkl --emb-file output/facebook_DynRWSG_threshold_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.2 --emb-dim 128 --workers 40  >> bash/log/facebook_DynRWSG_threshold.txt;
echo '--done--';
echo '--done--' >> bash/log/facebook_DynRWSG_threshold.txt;

echo '-------------------update-threshold 0.25 -----------------------------------------' >> bash/log/facebook_DynRWSG_threshold.txt;
echo '-------------------update-threshold 0.25 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/facebook/facebook.pkl --emb-file output/facebook_DynRWSG_threshold_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.25 --emb-dim 128 --workers 40  >> bash/log/facebook_DynRWSG_threshold.txt;
echo '--done--';
echo '--done--' >> bash/log/facebook_DynRWSG_threshold.txt;

echo '-------------------update-threshold 0.3 -----------------------------------------' >> bash/log/facebook_DynRWSG_threshold.txt;
echo '-------------------update-threshold 0.3 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/facebook/facebook.pkl --emb-file output/facebook_DynRWSG_threshold_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.3 --emb-dim 128 --workers 40  >> bash/log/facebook_DynRWSG_threshold.txt;
echo '--done--';
echo '--done--' >> bash/log/facebook_DynRWSG_threshold.txt;

echo '-------------------update-threshold 0.4 -----------------------------------------' >> bash/log/facebook_DynRWSG_threshold.txt;
echo '-------------------update-threshold 0.4 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/facebook/facebook.pkl --emb-file output/facebook_DynRWSG_threshold_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.4 --emb-dim 128 --workers 40  >> bash/log/facebook_DynRWSG_threshold.txt;
echo '--done--';
echo '--done--' >> bash/log/facebook_DynRWSG_threshold.txt;

echo '-------------------update-threshold 0.6 -----------------------------------------' >> bash/log/facebook_DynRWSG_threshold.txt;
echo '-------------------update-threshold 0.6 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/facebook/facebook.pkl --emb-file output/facebook_DynRWSG_threshold_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.6 --emb-dim 128 --workers 40  >> bash/log/facebook_DynRWSG_threshold.txt;
echo '--done--';
echo '--done--' >> bash/log/facebook_DynRWSG_threshold.txt;

echo '-------------------update-threshold 0.8 -----------------------------------------' >> bash/log/facebook_DynRWSG_threshold.txt;
echo '-------------------update-threshold 0.8 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/facebook/facebook.pkl --emb-file output/facebook_DynRWSG_threshold_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.8 --emb-dim 128 --workers 40  >> bash/log/facebook_DynRWSG_threshold.txt;
echo '--done--';
echo '--done--' >> bash/log/facebook_DynRWSG_threshold.txt;

echo '-------------------update-threshold 1.0 -----------------------------------------' >> bash/log/facebook_DynRWSG_threshold.txt;
echo '-------------------update-threshold 1.0 -----------------------------------------';
python src/main.py --method DynRWSG --task lp --graph data/facebook/facebook.pkl --emb-file output/facebook_DynRWSG_threshold_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 1.0 --emb-dim 128 --workers 40  >> bash/log/facebook_DynRWSG_threshold.txt;
echo '--done--';
echo '--done--' >> bash/log/facebook_DynRWSG_threshold.txt;

echo '......well done......' >> bash/log/facebook_DynRWSG_threshold.txt;