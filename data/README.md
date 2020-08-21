# About dataset
Here we provide three relatively small datasets used in the experiments, due to the space limit in Github. 
For other three datasets, one can follow the paper "GloDyNE: Global Topology Preserving Dynamic Network Embedding" to construct their dynamic networks.

Concretely, the format of an input dynamic network is prepared as follows: <br>
1) create an empty dynamic network as a empty python list, called DynG; <br>
2) use Networkx to build the graph based on the edge steams from time step t-k (here we let t-k=0) to t, called Gt; <br>
3) append the current snapshot Gt to the dynamic network DynG; <br>
4) repeat 2) and 3) as time goes on... <br>
5) finally, use Pickle to store DynG as a .pkl file

Note that, we take out the Largest Connected Component (LCC) of the very first snapshot G0. Based on the LCC of G0, we then develop G1, G2, ... such that every snapshot is a connected graph.

Besides, the raw dataset in http://konect.uni-koblenz.de is no longer available. If you would like to access the dataset, please check http://konect.cc/networks. And if you need the source code for data preprocessing, please Email me.