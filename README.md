# DynNE-RWSG
### Dynamic Network Embedding: An Approach based on Random Walks and Skip-Gram

by Chengbin HOU & Han ZHANG 2019 @ University of Birmingham

## Usages (testing in CMD)
#### Requirements
```bash
cd DynNE
pip install -r requirements.txt
```
Python 3.6.6 or above is required due to the new [`print(f' ')`](https://docs.python.org/3.6/reference/lexical_analysis.html#f-strings) feature
#### To obtain node embeddings as well as evaluate the quality
```bash
cd DynNE
python src/main.py --method DynRWSG --task all --graph data/cora/cora_dyn_graphs.pkl --label data/cora/cora_node_label_dict.pkl --emb-file output/cora_DynRWSG_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.1 --emb-dim 128 --workers 24
```
#### To save, and then load and evaluate node embeddings in different downstream tasks
```bash
cd DynNE
python src/main.py --method DynRWSG --task save --graph data/cora/cora_dyn_graphs.pkl --label data/cora/cora_node_label_dict.pkl --emb-file output/cora_DynRWSG_128_embs.pkl --num-walks 20 --restart-prob 0.2 --update-threshold 0.1 --emb-dim 128 --workers 24
python src/eval.py --task all --graph data/cora/cora_dyn_graphs.pkl --label data/cora/cora_node_label_dict.pkl --emb-file output/cora_DynRWSG_128_embs.pkl
```
#### To have an intuitive feeling of node embeddings in 2D/3D [todo...]
```bash
cd DynNE
python src/vis.py --emb-file output/cora_abrw_emb --label-file data/cora/cora_label.txt
```

## Usages (advanced, for batch testing)
```bash
cd DynNE
bash bash/cora_DynRWSG.sh
```
## Datasets
We take the latest 11 time steps (or 10 time intervals) of the following dynamic networks:

DynFacebook (social): 

DynHepth (citation):

DynAS733 (Anomymous System):

DynLFR (synthetic):