# DynWalks
A dynamic network embedding method with some desirable properties:
<br> --- Global Topology and Recent Changes Awareness
<br> --- Excellent Time and space efficiency
<br> --- Fulfilling real-time constraint if needed
<br> --- Handling unseen nodes without placeholders or knowing them in advance

This repository is for reproducing the results in paper <br>
"DynWalks: Global Topology and Recent Changes Awareness Dynamic Network Embedding" <br>
https://arxiv.org/abs/1907.11968

If you find it useful, please use the following citation.
```
@article{hou2019dynwalks,
    title={DynWalks: Global Topology and Recent Changes Awareness Dynamic Network Embedding},
    author={Chengbin Hou and Han Zhang and Ke Tang and Shan He},
    journal={arXiv preprint arXiv:1907.11968},
    year={2019}
}
```

## Usage (testing in CMD)
#### Requirement
```bash
cd DynNE
pip install -r requirements.txt
```
Python 3.6.6 or above is required due to the new [`print(f' ')`](https://docs.python.org/3.6/reference/lexical_analysis.html#f-strings) feature
#### To obtain node embeddings as well as evaluate the quality
```bash
cd DynNE
python src/main.py --method DynWalks --task all --graph data/cora/cora_dyn_graphs.pkl --emb-file output/cora_DynWalks_embs.pkl --scheme 3 --limit 0.2 --local-global 0.5 --num-walks 20 --walk-length 80 --window 10 --emb-dim 128 --workers 6
```
#### To save, and then load and evaluate node embeddings for different downstream tasks
```bash
cd DynNE
python src/main.py --method DynWalks --task save --graph data/cora/cora_dyn_graphs.pkl --emb-file output/cora_DynWalks_embs.pkl --scheme 3 --limit 0.2 --local-global 0.5 --num-walks 20 --walk-length 80 --window 10 --emb-dim 128 --workers 6
python src/eval.py --task all --graph data/cora/cora_dyn_graphs.pkl --emb-file output/cora_DynWalks_128_embs.pkl
```

## Usage (advanced, for batch testing)
```bash
cd DynNE
bash bash/ALL.sh
```

## Datasets
Please see the [README.md](https://github.com/houchengbin/DynWalks/tree/master/data) under **data folder**, in which all the data preprocessing 'py' files are provided, as well as the hyperlinks to original datasets.

## Issues
We are happy to answer any questions about the code and paper.