# GloDyNE
The aim of this work is to propose an efficient **dynamic network embedding** method for the **better global topology preservation** of a dynamic network at each time step. Unlike all previous works that mainly consider the most affected regions of a network, the idea of this work is to partition a network into smaller sub-networks so that we can **diversely** consider the topological changes over a network. <br>

Please refer to our [preprint](https://arxiv.org/abs/2008.01935) or [IEEE-TKDE early access version](https://ieeexplore.ieee.org/document/9302718) for further details. If you find this work is useful, please use the following citation.
```
@article{hou2020glodyne,
    title={GloDyNE: Global Topology Preserving Dynamic Network Embedding},
    author={Hou, Chengbin and Zhang, Han and He, Shan and Tang, Ke},
    journal={IEEE Transactions on Knowledge and Data Engineering},
    year={2020},
    doi={10.1109/TKDE.2020.3046511}
}
```

The motivation of this work is that any changes, i.e., edges being added or deleted, would affect all nodes in a connected network and greatly modify the proximity between nodes over a network via the high-order proximity as illustrated in Figure Fig a-c). On the other hand, as observed in Figure Fig1 d-f), the real-world dynamic networks usually have some inactive sub-networks where no change occurs lasting for several time steps. Putting both together, the existing DNE methods that focus on the most affected nodes (belonging to the active sub-networks) but do not consider the inactive sub-networks, would overlook the accumulated topological changes propagating to the inactive sub-networks via the high-order proximity. However, previous works did not consider this issue.

<center>
    <img src="https://github.com/houchengbin/GloDyNE/blob/master/data/Fig1.jpg" width="800"/>
</center>

Fig. a) A change (new edge in red) affects all nodes in the connected network via high-order proximity. The proximity of nodes 1-6 becomes $1^{st}$ order from $5^{th}$ order, nodes 2-6 becomes $2^{nd}$ order from $4^{th}$ order, etc. The proximity of any node in sub-network 1 to any node in sub-network 2 is reduced by 5 orders. b-c) How to calculate the modifications of the proximity between two snapshots, and the results show the modifications caused by a single edge can be very large in the real-world dynamic networks. d-f) The real-world dynamic networks have some inactive sub-networks, e.g., defined as no change occurs lasting for at least 5 time steps. The sub-networks, each of which has about 50 nodes, are obtained by applying METIS algorithm [Karypis and Kumar 1998] on the largest snapshot of a dynamic network. The details of the three dynamic networks are described in Section 5.1.1.

## Requirement
```bash
conda create -n GloDyNE python=3.6.8
source activate GloDyNE
cd GloDyNE
pip install -r requirements.txt
```
You may also need to **install [METIS package](https://github.com/networkx/networkx-metis)** from the source. <br>
Python 3.6.6 or above is required due to the new [`print(f' ')`](https://docs.python.org/3.6/reference/lexical_analysis.html#f-strings) feature.

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
If you would like to use your own dataset, please see the [README.md under the data folder](https://github.com/houchengbin/DynWalks/tree/master/data).

## Issues
We are happy to answer any questions about the code and paper.

## About DynWalks
If you are more interested in the idea about "freely trade-off between global topology and recent changes", you could check our preprint (for DynWalks method) at https://arxiv.org/abs/1907.11968 <br>
Since the general framework for DynWalks and GloDyNE is the same, you may either use the above citation for GloDyNE (we recommend this one as it will go through the peer review process), or use the following citation if you think it is more appropriate. 
```
@article{hou2019dynwalks,
    title={DynWalks: Global Topology and Recent Changes Awareness Dynamic Network Embedding},
    author={Chengbin Hou and Han Zhang and Ke Tang and Shan He},
    journal={arXiv preprint arXiv:1907.11968},
    year={2019}
}
```
To reproduce the results in DynWalks, please see v0.1 at https://github.com/houchengbin/GloDyNE/releases/tag/v0.1
