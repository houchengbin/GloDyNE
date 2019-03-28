'''
A Dynamic Network Embedding Method: DynWalks
by Chengbin HOU & Han ZHANG 2019

---------------------------------
G0: graph at previous time step
G1: graph at current time step
---------------------------------

todo:
1) reset deledted nodes or continue traning
2) what about the deleted nodes while doing negative sampling
3) currently accept fixed size of dynamic graphs; but it would be better to accept streaming graphs/edges;
     OR accept previous DynWalks model and continue traning
4) parallel random walk


# Regarding the novelty, we may need focus on the following points------------------------------------------------------------------------------------------------------------------------
# Method 1 -------- our novelty depends on 1) and 2)
# 1) how to select m most affected nodes -> further reduce complexity without lossing too much accuracy (by considering accumulated diff in some kind of reservoir using degree or else)
# 2) how to random walk start from those m nodes -> random walks with restart, if prob=0 -> naive random walk; if prob=1 -> always stay at starting node; try to tune prob
# 3) once obtain sequences for each of m nodes, shall we directly update OR reset them and then update? [do not waste too much time]
# 4) the hyper-parameters of Word2Vec SGNE model especially window, negative, iter [do not waste too much time]
# 5) when to restart? I think we only need to propose our general offline and online framework. when to restart is out futher work... [ignore for a moment]

# Method 2 -------- [ignore for a moment] [ignore for a moment] [ignore for a moment]
# 1) based on the diff between G1 and G0 -> added and deleted edges/nodes
# 2) way1: based on the changed edges/nodes -> handcraft sequences/sentences -> feed to Word2Vec [idea: synthetic sequences; still based on Python Gensim]
# 2) way2: feed the changed edges/nodes -> feed to SGNS [idea: code SGNE from scratch; using Python TensorFlow]
'''

import time
import random
import pickle
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
import logging
import numpy as np
import networkx as nx

from .utils import edge_s1_minus_s0, unique_nodes_from_edge_set


class DynWalks(object):
     def __init__(self, G_dynamic, restart_prob=0.2, update_threshold=0.1, emb_dim=128, num_walks=20, walk_length=80, 
                    window=10, workers=20, negative=5, seed=2019, limit=0.1, scheme=3):
          self.G_dynamic = G_dynamic.copy()  # a series of dynamic graphs
          self.emb_dim = emb_dim   # node emb dimensionarity
          self.update_threshold = update_threshold # NOT used anymore, as limit will automatically adjust update_threshold [will be deleted later]
          self.restart_prob = restart_prob  # restart probability for random walks
          self.num_walks = num_walks  # num of walks start from each node
          self.walk_length = walk_length  # walk length for each walk
          self.window = window  # Skip-Gram parameter
          self.workers = workers  # Skip-Gram parameter
          self.negative = negative  # Skip-Gram parameter
          self.seed = seed  # Skip-Gram parameter

          self.scheme = scheme
          self.limit = limit

          self.emb_dicts = [] # emb_dict @ t0, t1, ...; len(self.emb_dicts) == len(self.G_dynamic)
          self.reservoir = {} # {nodeID: num of affected, ...}
          
     def sampling_traning(self):
          # SGNS and suggested parameters to be tuned: size, window, negative, workers, seed
          # to tune other parameters, please read https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
          w2v = gensim.models.Word2Vec(sentences=None, size=self.emb_dim, window=self.window, sg=1, hs=0, negative=self.negative, ns_exponent=0.75,
                              alpha=0.025, min_alpha=0.0001, min_count=1, sample=0.001, iter=4, workers=self.workers, seed=self.seed,
                              corpus_file=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                              max_vocab_size=None, max_final_vocab=None, trim_rule=None)  # w2v constructor, default parameters
     
          for t in range(len(self.G_dynamic)):
               t1 = time.time()
               if t ==0:  # offline ----------------------------
                    G0 = self.G_dynamic[t]    # initial graph
                    sentences = simulate_walks(nx_graph=G0, num_walks=self.num_walks, walk_length=self.walk_length, restart_prob=None) #restart_prob=None or 0 --> deepwalk
                    sentences = [[str(j) for j in i] for i in sentences]
                    w2v.build_vocab(sentences=sentences, update=False) # init traning, so update False
                    w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter) # follow w2v constructor
               else:      # online adapting --------------------
                    G0 = self.G_dynamic[t-1]  # previous graph
                    G1 = self.G_dynamic[t]    # current graph
                    node_update_list, self.reservoir = node_selecting_scheme(graph_t0=G0, graph_t1=G1, reservoir_dict=self.reservoir, limit=self.limit, scheme=self.scheme)
                    sentences = simulate_walks(nx_graph=G1, num_walks=self.num_walks, walk_length=self.walk_length, restart_prob=self.restart_prob, affected_nodes=node_update_list)
                    sentences = [[str(j) for j in i] for i in sentences]
                    w2v.build_vocab(sentences=sentences, update=True) # online update
                    w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter)

               emb_dict = {} # {nodeID: emb_vector, ...}
               for node in self.G_dynamic[t].nodes():
                    emb_dict[node] = w2v.wv[str(node)]
               self.emb_dicts.append(emb_dict)
               t2 = time.time()
               print(f'DynWalks sampling and traning time: {(t2-t1):.2f}s --> {t+1}/{len(self.G_dynamic)}')
          
          return self.emb_dicts  # To save memory useage, we can delete DynWalks model after training

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




# ----------------------------------------------------------------------------------------------------
# ---------- utils: most_affected_nodes, simulate_walks, random_walk, random_walk_restart ------------
# ----------------------------------------------------------------------------------------------------

def node_selecting_scheme(graph_t0, graph_t1, reservoir_dict, limit=0.1, scheme=3):
     ''' select nodes to be updated
          G0: previous graph @ t-1;
          G1: current graph  @ t;
          reservoir_dict: will be always maintained in ROM
          limit: fix the number of node --> the percentage of nodes of a network to be updated (exclude new nodes)

          scheme 0: new nodes (DeppWalk-SGNE dynamic version)
          scheme 1: new nodes + random nodes 
          scheme 2: new nodes + most affected nodes + random nodes
          scheme 3: new nodes + most affected nodes + random nodes (much diverse?; also exclude most affected their neighbors) 
          scheme 4: new nodes + most affected nodes + random nodes that will be selected if with large degree
          scheme 5: new nodes + most affected nodes + random nodes that will be selected if with small degree 
     '''
     G0 = graph_t0.copy()
     G1 = graph_t1.copy()
     edge_add = edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges())) # one may directly use streaming added edges if possible
     edge_del = edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges())) # one may directly use streaming added edges if possible

     node_affected_by_edge_add = unique_nodes_from_edge_set(edge_add) # unique
     node_affected_by_edge_del = unique_nodes_from_edge_set(edge_del) # unique
     node_affected = list(set(node_affected_by_edge_add + node_affected_by_edge_del)) # unique
     node_add = [node for node in node_affected_by_edge_add if node not in G0.nodes()]
     node_del = [node for node in node_affected_by_edge_del if node not in G1.nodes()]
     exist_node_affected = list(set(node_affected) - set(node_add) - set(node_del))  # affected nodes are in both G0 and G1

     t1 = time.time()
     # for fair comparsion, the number of nodes to be updated are the same for schemes 1, 2, 3, 4, 5 whereas scheme 0 is DeepWalk-SGNE
     num_limit = int(G1.number_of_nodes() * limit)
     print('num_limit', num_limit)
     # remain some positions for random nodes to increase diversity for preserving global network structure
     # num_limit_half = int(num_limit * 0.5)
     # choose the top "num_limit_half" most affected nodes for preserving the local structure of most affected nodes
     most_affected_nodes, reservoir_dict = select_most_affected_nodes(G0, G1, num_limit, reservoir_dict, exist_node_affected)

     node_update_list = []   # all the nodes to be updated 
     if scheme == 0:
          print('scheme == 0')
          node_update_list = node_add

     if scheme == 1:
          print('scheme == 1')
          tabu_nodes = node_add
          all_nodes = [node for node in G1.nodes() if node not in tabu_nodes]
          num_limit_random = num_limit
          random_nodes = list(np.random.choice(all_nodes, num_limit_random, replace=False))
          node_update_list = random_nodes + node_add

     if scheme == 2:
          print('scheme == 2')
          tabu_nodes = list(set(node_add + most_affected_nodes))      # different from scheme 1, we add some most affected nodes
          all_nodes = [node for node in G1.nodes() if node not in tabu_nodes]
          num_limit_random = num_limit-len(most_affected_nodes)
          random_nodes = list(np.random.choice(all_nodes, num_limit_random, replace=False))
          node_update_list = random_nodes + node_add + most_affected_nodes

     if scheme == 3:
          print('scheme == 3')
          most_affected_nbrs = []
          for node in most_affected_nodes:
               most_affected_nbrs.extend( list(nx.neighbors(G=G1, n=node)) )           # what about nbrs of nbrs? more diversity!
          tabu_nodes = list(set(node_add + most_affected_nodes + most_affected_nbrs))  # different from scheme 2, we further increase diversity
          all_nodes = [node for node in G1.nodes() if node not in tabu_nodes]
          num_limit_random = num_limit-len(most_affected_nodes)
          random_nodes = list(np.random.choice(all_nodes, num_limit_random, replace=False))
          node_update_list = random_nodes + node_add + most_affected_nodes

     if scheme == 4:
          print('scheme == 4')
          tabu_nodes = list(set(node_add + most_affected_nodes))
          all_nodes = [node for node in G1.nodes() if node not in tabu_nodes]
          all_nodes_degrees = [G1.degree[node] for node in all_nodes]
          degree_dist = np.array(all_nodes_degrees) / np.array(all_nodes_degrees).sum()  #more likely to choose node with larger degree
          num_limit_random = num_limit-len(most_affected_nodes)
          random_nodes = list(np.random.choice(all_nodes, num_limit_random, replace=False, p=degree_dist))
          node_update_list = random_nodes + node_add + most_affected_nodes

     if scheme == 5:
          print('scheme == 5')
          tabu_nodes = list(set(node_add + most_affected_nodes))
          all_nodes = [node for node in G1.nodes() if node not in tabu_nodes]
          all_nodes_degrees = [G1.degree[node] for node in all_nodes]
          inverse_all_nodes_degrees = 1.0 / np.array(all_nodes_degrees)
          degree_dist = np.array(inverse_all_nodes_degrees) / np.array(inverse_all_nodes_degrees).sum() #more likely to choose node with smaller degree
          num_limit_random = num_limit-len(most_affected_nodes)
          random_nodes = list(np.random.choice(all_nodes, num_limit_random, replace=False, p=degree_dist))
          node_update_list = random_nodes + node_add + most_affected_nodes
     
     reservoir_key_list = list(reservoir_dict.keys())
     for node in node_update_list:
          if node in reservoir_key_list:
               del reservoir_dict[node]  # if updated, delete it

     t2 = time.time()
     print(f'--> node selecting time; time cost: {(t2-t1):.2f}s')
     print(f'num of nodes in reservoir with accumulated changes but not updated {len(list(reservoir_dict))}')
     print(f'# nodes added {len(node_add)}, # nodes deleted {len(node_del)}, # nodes updated {len(node_update_list)}')
     print(f'# nodes affected {len(node_affected)}, # nodes most affected {len(most_affected_nodes)}')
     return node_update_list, reservoir_dict


def select_most_affected_nodes(G0, G1, num_limit_return_nodes, reservoir_dict, exist_node_affected):
     ''' return num_limit_return_nodes to be updated
          based on the ranking of the accumulated changes w.r.t. their local connectivity
     '''
     most_affected_nodes = []
     for node in exist_node_affected:
          nbrs_set1 = set(nx.neighbors(G=G1, n=node))
          nbrs_set0 = set(nx.neighbors(G=G0, n=node))
          changes = len( nbrs_set1.union(nbrs_set0) - nbrs_set1.intersection(nbrs_set0) )
          if node in reservoir_dict.keys():
               reservoir_dict[node] += changes # accumulated changes
          else:
               reservoir_dict[node] = changes  # newly added changes
               
     if len(exist_node_affected) > num_limit_return_nodes:
          reservoir_dict_score = {}
          for node in exist_node_affected:
               reservoir_dict_score[node] = reservoir_dict[node] / G0.degree[node]
          # worse case O(n) https://docs.scipy.org/doc/numpy/reference/generated/numpy.partition.html
          # the largest change at num_limit_return_nodes will be returned
          cutoff_score = np.partition(list(reservoir_dict_score.values()), -num_limit_return_nodes, kind='introselect')[-num_limit_return_nodes]
          cnt = 0
          for node in reservoir_dict_score.keys():
               if reservoir_dict_score[node] >= cutoff_score: # fix bug: there might be multiple equal cutoff_score nodes...
                    most_affected_nodes.append(node)
                    cnt += 1
                    if cnt == num_limit_return_nodes:         # fix bug: we need exactly the number of limit return nodes...
                         break
     else: # NOTE: if most_affected_nodes are less than half_limit, additional random nodes will be automatically sampled for compensation
          most_affected_nodes = exist_node_affected
     return most_affected_nodes, reservoir_dict

def select_most_affected_nodes_nbrs(G1, most_affected_nodes):
     most_affected_nbrs = []
     for node in most_affected_nodes:
          most_affected_nbrs.extend( list(nx.neighbors(G=G1, n=node)) )
     return list(set(most_affected_nbrs)) #return a list without repeated items


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
     
     ''' multi-processors; use it iff the # of nodes over 20k
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