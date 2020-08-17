"""
The proposed methoed: GloDyNE
---------------------------------
limit=0.1                          # limited computational resources i.e. the upper limit # of selected nodes
                                   # NOTE: limit i.e. $\alpha$ in our paper
num_walks=10, walk_length=80,      # random walk hyper-parameters
window=10, negative=5,             # Skip-Gram hyper-parameters
seed=2019, workers=32,             # others
G0                                 # snapshot @t-1
G1                                 # snapshot @t
---------------------------------
by Chengbin Hou & Han Zhang @ 2020
"""

import time
import random
import pickle
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
import logging
import numpy as np
import networkx as nx
import math

from .utils import edge_s1_minus_s0, unique_nodes_from_edge_set

# ===============================================================================================================================
# ======================== CORE1: Overall framework to **incrementally** learn node embeddings ==================================
# ===============================================================================================================================
class DynWalks(object):
     def __init__(self, G_dynamic, limit, num_walks, walk_length, window, emb_dim, negative, workers, seed, scheme):
          self.G_dynamic = G_dynamic.copy()  # a series of dynamic graphs
          self.emb_dim = emb_dim             # node emb dimensionarity
          self.num_walks = num_walks         # num of walks start from each node
          self.walk_length = walk_length     # walk length for each walk
          self.window = window               # Skip-Gram parameter
          self.workers = workers             # Skip-Gram parameter
          self.negative = negative           # Skip-Gram parameter
          self.seed = seed                   # Skip-Gram parameter
          self.scheme = scheme
          self.limit = limit

          self.emb_dicts = [] # emb_dict @ t0, t1, ...; len(self.emb_dicts) == len(self.G_dynamic)
          self.reservoir = {} # {nodeID: num of changes, ...}
          
     def sampling_traning(self):
          # SGNS and suggested parameters to be tuned: size, window, negative, workers, seed
          # to tune other parameters, please read https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
          w2v = gensim.models.Word2Vec(sentences=None, size=self.emb_dim, window=self.window, sg=1, hs=0, negative=self.negative, ns_exponent=0.75,
                              alpha=0.025, min_alpha=0.0001, min_count=1, sample=0.001, iter=4, workers=self.workers, seed=self.seed,
                              corpus_file=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                              max_vocab_size=None, max_final_vocab=None, trim_rule=None)  # w2v constructor, default parameters
     
          for t in range(len(self.G_dynamic)):
               t1 = time.time()
               if t == 0:   # offline ==============================
                    G0 = self.G_dynamic[t]    # initial graph
                    sentences = simulate_walks(nx_graph=G0, num_walks=self.num_walks, walk_length=self.walk_length)
                    sentences = [[str(j) for j in i] for i in sentences]
                    w2v.build_vocab(sentences=sentences, update=False) # init traning, so update False
                    w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter) # follow w2v constructor

               else:   # incremental adapting ==============================
                    G0 = self.G_dynamic[t-1]  # previous graph
                    G1 = self.G_dynamic[t]    # current graph
                    node_update_list, self.reservoir = node_selecting_scheme(graph_t0=G0, graph_t1=G1, reservoir_dict=self.reservoir, limit=self.limit, scheme=self.scheme)
                    # print(node_update_list)
                    # node_update_list_2_txt(node_update_list,'node_update_list.txt')
                    sentences = simulate_walks(nx_graph=G1, num_walks=self.num_walks, walk_length=self.walk_length, affected_nodes=node_update_list)
                    # sentences_2_pkl(sentences,'sentences.pkl')
                    # with open('sentences.pkl', 'rb') as f:
                    #     any_object = pickle.load(f)
                    sentences = [[str(j) for j in i] for i in sentences]
                    w2v.build_vocab(sentences=sentences, update=True)     # incremental update
                    w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter)

               emb_dict = {}    # {nodeID: emb_vector, ...}
               for node in self.G_dynamic[t].nodes():
                    emb_dict[node] = w2v.wv[str(node)]
               self.emb_dicts.append(emb_dict)
               t2 = time.time()
               print(f'DynWalks sampling and traning time: {(t2-t1):.2f}s --> {t+1}/{len(self.G_dynamic)}')  
          return self.emb_dicts

     def save_emb(self, path='unnamed_dyn_emb_dicts.pkl'):
          ''' save # emb_dict @ t0, t1, ... to a file using pickle
          '''
          with open(path, 'wb') as f:
               pickle.dump(self.emb_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
     
     def load_emb(self, path='unnamed_dyn_emb_dicts.pkl'):
          ''' load # emb_dict @ t0, t1, ... to a file using pickle
          '''
          with open(path, 'rb') as f:
               any_object = pickle.load(f)
          return any_object


# ===== update reservoir dict {node_ID: changes, ...} based on the steam edges ===== 
def update_reservoir_dict(G0, G1, reservoir_dict, exist_node_affected):
     for node in exist_node_affected:
          nbrs_set1 = set(nx.neighbors(G=G1, n=node))
          nbrs_set0 = set(nx.neighbors(G=G0, n=node))
          changes = len( nbrs_set1.union(nbrs_set0) - nbrs_set1.intersection(nbrs_set0) ) # steam edges
          if node in reservoir_dict.keys():
               reservoir_dict[node] += changes # accumulated changes if already in reservoir
          else:
               reservoir_dict[node] = changes  # newly added changes if not in reservoir
     return reservoir_dict


# ==================================================================================================================================================
# ================================================= CORE2: online node selecting strategy ==========================================================
# ==================================================================================================================================================
def node_selecting_scheme(graph_t0, graph_t1, reservoir_dict, limit=0.1, scheme=4):   # currently, only focus on the changes of network **topology**
     ''' select nodes to be updated
          G0: previous graph @ t-1;
          G1: current graph  @ t;
          reservoir_dict: will be always maintained in ROM
          limit: fix the number of node --> the percentage of nodes of a network to be updated (exclude new nodes)
          scheme 4 for METIS based node selecting approach; scheme 1-3 for other approaches
     '''
     G0 = graph_t0.copy()
     G1 = graph_t1.copy()
     edge_add = edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges()))  # one may directly use steam added edges if possible
     edge_del = edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges()))

     node_affected_by_edge_add = unique_nodes_from_edge_set(edge_add)
     node_affected_by_edge_del = unique_nodes_from_edge_set(edge_del) 
     node_affected = list(set(node_affected_by_edge_add + node_affected_by_edge_del)) 
     node_add = [node for node in node_affected_by_edge_add if node not in G0.nodes()]
     node_del = [node for node in node_affected_by_edge_del if node not in G1.nodes()]
     
     exist_node_affected = list(set(node_affected) - set(node_add) - set(node_del))           # now, we only consider the 1st-order affected nodes are in both G0 and G1; 
     exist_node_not_affected = list(set(G1.nodes())- set(node_add)-set(exist_node_affected))  # for 2nd-order, see "select_most_affected_nodes_nbrs"

     if len(node_del) !=0:
          reservoir_key_list = list(reservoir_dict.keys())
          for node in node_del:
               if node in reservoir_key_list:
                    del reservoir_dict[node]  # if a node is deleted, also delete it from reservoir

     t1 = time.time()
     num_limit = int(G1.number_of_nodes() * limit)   # the maximum number of nodes to be selected i.e. **alpha** in the paper
     most_affected_nodes = []  # used in scheme 1
     random_nodes = []         # used in scheme 2
     diverse_nodes = []        # used in scheme 3 and scheme 4
     node_update_list = []     # all the nodes to be updated

     reservoir_dict = update_reservoir_dict(G0, G1, reservoir_dict, exist_node_affected)  # update reservoir dict {node_ID: changes, ...} based on the steam edges

     #----------------------------------------------------------------------------------------------------------------- node selecting strategy 4
     #NOTE: one may use different node selecting strategy, so that other desireable network topology can be encoded into random walks
     if True:    
          print('scheme == 4, the METIS based diverse approach biased to most affected nodes')
          import nxmetis
          start_comm_det = time.time()
          cost_parts = nxmetis.partition(G=G1, nparts=num_limit)
          parts = cost_parts[1]        # cost = cost_parts[0] useless
          empty_part_counter = 0
          for part in parts:           # part i.e. community, operation in one community at each loop
               if len(part) == 0:
                    empty_part_counter += 1
               else:
                    node_scores = []                # node_scores within this part
                    for node in part:
                         try:
                              node_scores.append(math.exp(reservoir_dict[node]/G0.degree[node]))
                         except:
                              node_scores.append(1) # (2 or e)^0 = 1
                    node_scores_prob = []           # normalize node_scores within this part
                    part_sum = sum(node_scores) 
                    for i in range(len(node_scores)):
                         node_scores_prob.append(node_scores[i]/part_sum)
                    # sample one node from this part based on node_scores_prob, which bias to recent changes
                    diverse_nodes.append( np.random.choice(part, p=node_scores_prob) )
          
          # ---- due to the limitation of METIS, there might be few empty parts ----
          if empty_part_counter != 0:
               remaining_pool = list(G1.nodes()- set(node_add) - set(diverse_nodes))
               remaining_pool_score = []
               for node in remaining_pool:
                         try:
                              remaining_pool_score.append(math.exp(reservoir_dict[node]/G0.degree[node]))
                         except:
                              remaining_pool_score.append(1)
               remaining_pool_score_sum = sum(remaining_pool_score)
               remaining_pool_scores_prob = []
               for i in range(len(remaining_pool_score)):
                         remaining_pool_scores_prob.append(remaining_pool_score[i]/remaining_pool_score_sum)
               diverse_nodes.extend( np.random.choice(remaining_pool, size=empty_part_counter, replace=True,  p=remaining_pool_scores_prob) )
          end_comm_det = time.time()
          print('MIETS time: ', end_comm_det-start_comm_det)
          node_update_list =  node_add + diverse_nodes
     #----------------------------------------------------------------------------------------------------------------- END of node selecting strategy 4

     for node in node_update_list:
          try:
               del reservoir_dict[node]  # if updated, delete it from reservoir
          except:
               pass
     t2 = time.time()
     print(f'--> node selecting time; time cost: {(t2-t1):.2f}s')
     print(f'# num_limit {num_limit}, # nodes updated {len(node_update_list)}')
     print(f'# nodes added {len(node_add)}, # nodes deleted {len(node_del)}')
     print(f'# nodes most affected {len(most_affected_nodes)}  \t ===== S1 =====')
     print(f'# of random nodes {len(random_nodes)}         \t ===== S2 =====')
     print(f'# diverse nodes {len(diverse_nodes)}        \t ===== S3 or S4 =====')
     print(f'# nodes in reservoir with accumulated changes but not updated {len(list(reservoir_dict))}')
     print(f'# all nodes affected {len(node_affected)}')
     return node_update_list, reservoir_dict


# =================================================================================================  # todo... multiprocessors
# ===================================== CORE3: random walk sampling ===============================  # refer to https://github.com/houchengbin/OpenANE/blob/master/src/libnrl/walker.py
# =================================================================================================
def simulate_walks(nx_graph, num_walks, walk_length, restart_prob=None, affected_nodes=None):
     '''
     Repeatedly simulate random walks from each node
     '''
     G = nx_graph
     walks = []

     if affected_nodes == None: # simulate walks on every node in the graph [offline]
          nodes = list(G.nodes())
     else:                     # simulate walks on affected nodes [online]
          nodes = list(affected_nodes)
     
     ''' multi-processors; use it iff the # of nodes over 20k; if any bugs, refer to https://github.com/houchengbin/OpenANE/blob/master/src/libnrl/walker.py
     if restart_prob == None: # naive random walk
          t1 = time.time()
          for walk_iter in range(num_walks):
               random.shuffle(nodes)
               from itertools import repeat
               from multiprocessing import Pool, freeze_support
               with Pool(processes=5) as pool:
                    # results = [pool.apply_async(random_walk, args=(G, node, walk_length)) for node in nodes]
                    # results = [p.get() for p in results]
                    results = pool.starmap(random_walk, zip(repeat(G), nodes, repeat(walk_length)))
               for result in results:
                    walks.append(result)
          t2 = time.time()
          print('all walks',len(walks))
          print(f'random walk sampling, time cost: {(t2-t1):.2f}')
     '''
     if restart_prob == None: # naive random walk
          t1 = time.time()
          for walk_iter in range(num_walks):
               random.shuffle(nodes)
               for node in nodes:
                    walks.append(random_walk(nx_graph=G, start_node=node, walk_length=walk_length))
          t2 = time.time()
          print(f'random walk sampling, time cost: {(t2-t1):.2f}')
     else: # random walk with restart
          t1 = time.time()
          for walk_iter in range(num_walks):
               random.shuffle(nodes)
               for node in nodes:
                    walks.append(random_walk_restart(nx_graph=G, start_node=node, walk_length=walk_length, restart_prob=restart_prob))
          t2 = time.time()
          print(f'random walk sampling, time cost: {(t2-t1):.2f}')
     return walks

def random_walk(nx_graph, start_node, walk_length):
     '''
     Simulate a random walk starting from start node
     '''
     G = nx_graph
     walk = [start_node]

     while len(walk) < walk_length:
          cur = walk[-1]
          cur_nbrs = list(G.neighbors(cur))
          if len(cur_nbrs) > 0:
               walk.append(random.choice(cur_nbrs))
          else:
               break
     return walk

def random_walk_restart(nx_graph, start_node, walk_length, restart_prob):
     '''
     random walk with restart
     restart if p < restart_prob
     '''
     G = nx_graph
     walk = [start_node]

     while len(walk) < walk_length:
          p = random.uniform(0, 1)
          if p < restart_prob:
               cur = walk[0] # restart
               walk.append(cur)
          else:
               cur = walk[-1]
               cur_nbrs = list(G.neighbors(cur))
               if len(cur_nbrs) > 0:
                    walk.append(random.choice(cur_nbrs))
               else:
                    break
     return walk




# ========================================================================
# ============================== other utils =============================
# ========================================================================

def select_most_affected_nodes_nbrs(G1, most_affected_nodes):
     most_affected_nbrs = []
     for node in most_affected_nodes:
          most_affected_nbrs.extend( list(nx.neighbors(G=G1, n=node)) )
     return list(set(most_affected_nbrs))

def node_update_list_2_txt(my_list, path):
     with open(path, 'w') as f:
          for item in my_list:
               f.write("%s\n" % item)

def sentences_2_pkl(my_list, path):
     import collections
     new_list = []
     for items in my_list:
          for item in items:
               new_list.append(item)
     c = collections.Counter(new_list)

     with open(path, 'wb') as f:
          pickle.dump(c, f, protocol=pickle.HIGHEST_PROTOCOL)

def to_weighted_graph(graph):
     G = graph.copy()
     for e in G.edges():
          G[e[0]][e[1]]['weight'] = 1.0
     return G

def to_unweighted_graph(weighted_graph):
     pass
     # return unweighted_graph

"""
# ========= for S1 only =============
def select_most_affected_nodes(G0, G1, num_limit_return_nodes, reservoir_dict, exist_node_affected):
     ''' return num_limit_return_nodes to be updated
          based on the ranking of the accumulated changes w.r.t. their local connectivity
     '''
     most_affected_nodes = []
   
     if len(exist_node_affected) > num_limit_return_nodes:
          reservoir_dict_score = {}
          for node in exist_node_affected:
               try:
                    reservoir_dict_score[node] = reservoir_dict[node] / G0.degree[node]   # could be exp
               except:
                    print('G0.degree[node] = 0 for node: ', node, 'Degree is 0. Yes or No', G0.degree[node]==0)
                    # reservoir_dict_score[node] = reservoir_dict[node] / 0.1                # relatively very large changes  [公平起见，大家都忽略此点]
          # worse case O(n) https://docs.scipy.org/doc/numpy/reference/generated/numpy.partition.html
          # the largest change at num_limit_return_nodes will be returned
          cutoff_score = np.partition(list(reservoir_dict_score.values()), -num_limit_return_nodes, kind='introselect')[-num_limit_return_nodes]
          cnt = 0
          for node in reservoir_dict_score.keys():
               if reservoir_dict_score[node] >= cutoff_score: # fix bug: there might be multiple equal cutoff_score nodes...
                    if cnt == num_limit_return_nodes:         # fix bug: we need exactly the number of limit return nodes...
                         break
                    most_affected_nodes.append(node)
                    cnt += 1
     else:  #NOTE: len(exist_node_affected) <= num_limit_return_nodes
          most_affected_nodes = exist_node_affected
     return most_affected_nodes
# ========= for S3 only =============
def probabilistic_select_nodes_with_community_awareness(G0, G1, num_limit_return_nodes, reservoir_dict, node_add, comms):
     # ---- node_comms = {node: comm_ID, ...} ----
     node_comms = {}   # {node: comm_ID, ...}
     compressed_nodes = list(comms.keys())     # treat as comm_ID
     for comm_ID in compressed_nodes:
          comm_nodes_ids = comms[comm_ID]      # all nodes in a community
          for node_id in comm_nodes_ids:       # for each nodes in a community
               node_comms[node_id] = comm_ID   # {node: comm_ID, ...}
     '''
     for i in range(len(compressed_nodes)):
          comm_nodes_ids = comms[compressed_nodes[i]]              # all nodes in a community
          for j in range(len(comm_nodes_ids)):                     # for each nodes in a community
               node_comms[comm_nodes_ids[j]] = compressed_nodes[i] # {node: comm_ID, ...}
     '''
     # ---- com_prob = {community: prob, ...} ----
     com_prob = {}   # {community: prob, ...}
     comms_len = [len(comms[k]) for k in compressed_nodes]   # size of each community
     log_comms = []        # score for each community based on their size, use log for mapping
     for i in range(len(compressed_nodes)):
          log_comms.append(math.log2(comms_len[i]))  # base could be 2, e, etc. The larger base it is, the larger penalty on the large community size
     sum_log_comms = sum(log_comms)   # denominator for normalization
     for i in range(len(compressed_nodes)):
          com_prob[compressed_nodes[i]] = log_comms[i] / sum_log_comms
     # ---- reservoir_dict_score = {node: score, ...} ----
     reservoir_dict_score = {}   # {node: score, ...}
     all_node_sample_pool = list(set(G1.nodes()) - set(node_add))  # node_add has been sampled, so ignore them
     for node in all_node_sample_pool:
          try:
               reservoir_dict_score[node] = math.pow(2, reservoir_dict[node] / G0.degree[node])   # could be exp
          except:
               reservoir_dict_score[node] = 1 # (2 or e)^0 = 1
     # ---- comms_score_sum = {community: score_sum, ...} ----
     comms_score_sum = {}   # {community: score_sum, ...}
     for comm_ID in compressed_nodes:
          comms_score_sum[comm_ID] = 0   # inti to 0
     for node in all_node_sample_pool:
          current_node_score = reservoir_dict_score[node]
          current_node_comm_id = node_comms[node]
          comms_score_sum[current_node_comm_id] += current_node_score
     
     # ---- **community awareness biased to recent changes** probability distribution over all nodes in G1 (exclude new nodes) ----
     prob_all_node_sample_pool = []
     for node in all_node_sample_pool:
          current_node_comm_id = node_comms[node]
          current_comm_prob = com_prob[current_node_comm_id]                                         # current_comm_prob
          current_node_score = reservoir_dict_score[node]                                            
          current_node_prob_within_comm = current_node_score / comms_score_sum[current_node_comm_id] # current_node_prob_within_comm
          current_node_final_prob = current_comm_prob * current_node_prob_within_comm                # current_node_final_prob = current_comm_prob * current_node_prob_within_comm
          prob_all_node_sample_pool.append(current_node_final_prob)
     prob_all_node_sample_pool = np.array(prob_all_node_sample_pool) 
     prob_all_node_sample_pool /= prob_all_node_sample_pool.sum()     # re-normalize probability as there might be many round up loss in the process
     select_nodes = list(np.random.choice(all_node_sample_pool, num_limit_return_nodes, replace=True, p=prob_all_node_sample_pool))
     return select_nodes
"""