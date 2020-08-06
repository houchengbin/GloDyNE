# GloDyNE
The aim of this work is to propose an efficient **dynamic network embedding** method for **better global topology preservation** of a dynamic network at each time step. Unlike all previous works that mainly consider the most affected regions of a network, the idea of this work, motivated by divide and conquer, is to partition a network into smaller sub-networks such that we can **diversely** consider the topological changes over a network. <br>
The motivation of this work is that most real-world networks have some **inactivate regions/sub-networks** which would receive accumulated topological changes propagated via the high-order proximity. See the figure below for illustration. Therefore, in order to better preserve the global topology, we also need to consider the accumulated topological changes in the inactivate regions/sub-networks. However, previous works did not consider this issue.

<center>
    <img src="https://github.com/houchengbin/GloDyNE/blob/master/data/Fig1.jpg" width="666"/>
</center>

Fig. a) A change (new edge in red) affects all nodes in the connected network via high-order proximity. The proximity of nodes 1-6 becomes $1^{st}$ order from $5^{th}$ order, nodes 2-6 becomes $2^{nd}$ order from $4^{th}$ order, etc. Besides, the proximity of any node in sub-network 1 to any node in sub-network 2 is reduced by 5 orders. b-d) The real-world dynamic networks have some inactive sub-networks (e.g., defined as no change occurs lasting for at least 5 time steps). The x-axis indicates the number of consecutive time steps that no change occurs in a sub-network. The y-axis gives the counts of each case in x-axis. The sub-networks, in average 50 nodes per sub-network, are obtained by applying METIS algorithm [Karypis and Kumar 1998] on the largest snapshot of a dynamic network. The details of the three dynamic networks are described in Section 5.

Please refer to [our paper](https://arxiv.org/abs/2008.01935) for further details. If you find this work is useful, please use the following citation.
```
@misc{hou2020glodyne,
    title={GloDyNE: Global Topology Preserving Dynamic Network Embedding},
    author={Chengbin Hou and Han Zhang and Shan He and Ke Tang},
    year={2020},
    eprint={2008.01935},
    archivePrefix={arXiv},
    primaryClass={cs.SI}
}
```
Currently, we are preparing the revision for the second-round review to a journal based on the positive feedback.

## Requirement
```bash
conda create -n GloDyNE python=3.6.8
source activate GloDyNE
cd GloDyNE
pip install -r requirements.txt
```
You may also need to install [METIS package](https://github.com/networkx/networkx-metis) from the source. <br>
Note, Python 3.6.6 or above is required due to the new [`print(f' ')`](https://docs.python.org/3.6/reference/lexical_analysis.html#f-strings) feature.

## Usage
#### To obtain node embeddings as well as evaluate by graph reconstruction task
```bash
cd GloDyNE
python src/main.py --method DynWalks --task gr --graph data/AS733/AS733_new.pkl --label data/AS733/AS733_label.pkl --emb-file output/AS733_DynWalks.pkl --num-walks 10 --walk-length 80 --window 10 --limit 0.1 --scheme 4 --seed 2019 --emb-dim 128 --workers 32
```
#### To save, and then load and evaluate node embeddings for different downstream tasks
```bash
cd GloDyNE
python src/main.py --method DynWalks --task save --graph data/AS733/AS733_new.pkl --label data/AS733/AS733_label.pkl --emb-file output/AS733_DynWalks.pkl --num-walks 10 --walk-length 80 --window 10 --limit 0.1 --scheme 4 --seed 2019 --emb-dim 128 --workers 32
python src/eval.py --task all --graph data/AS733/AS733_new.pkl --emb-file output/AS733_DynWalks.pkl --label data/AS733/AS733_label.pkl --seed 2019
```
#### Advanced, for batch testing
```bash
cd GloDyNE
bash bash/ALL_small.sh
```

## Datasets
Please see the [README.md](https://github.com/houchengbin/DynWalks/tree/master/data) under the **data** folder. <br>
If you would like to use your own dataset, the input dynamic network can be prepared as follows: <br>
1) create an empty dynamic network as an empty python list, called DynG; <br>
2) use Networkx to build the graph based on the edge steams from time step t-k to t, called Gt; <br>
3) append the current snapshot Gt to the dynamic network DynG; <br>
4) repeat 2) and 3) as time goes on... <br>
5) finally, use Pickle to store DynG as a .pkl file

## Issues
We are happy to answer any questions about the code and paper.
