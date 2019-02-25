#! /bin/bash
echo '......Chengbin HOU @SUSTech 2019......';

echo '------------------- DynRWSG -----------------------------------------' > bash/log/LFR_DynRWSG.txt;
python src/main.py --method DynRWSG --task all --graph data/LFR/LFR.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_DynRWSG.txt;
echo '--done--' >> bash/log/LFR_DynRWSG.txt;
echo '--DynRWSG done--';

echo '------------------- DeepWalk -----------------------------------------' > bash/log/LFR_DeepWalk.txt;
python src/main.py --method DeepWalk --task all --graph data/LFR/LFR.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_DeepWalk.txt;
echo '--done--' >> bash/log/LFR_DeepWalk.txt;
echo '--DeepWalk done--';

echo '------------------- HOPE -----------------------------------------' > bash/log/LFR_HOPE.txt;
python src/main.py --method HOPE --task all --graph data/LFR/LFR.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_HOPE.txt;
echo '--done--' >> bash/log/LFR_HOPE.txt;
echo '--HOPE done--';

echo '------------------- GraRep -----------------------------------------' > bash/log/LFR_GraRep.txt;
python src/main.py --method GraRep --task all --graph data/LFR/LFR.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.1 --emb-dim 128 --workers 40  >> bash/log/LFR_GraRep.txt;
echo '--done--' >> bash/log/LFR_GraRep.txt;
echo '--GraRep done--';

echo '......well done......';