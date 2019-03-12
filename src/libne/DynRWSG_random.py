'''
add DynRWSG_random method
DynRWSG: update new nodes + selected most affected nodes from old nodes
DynRWSG_random: update new nodes + equal @ of DynRWSG selected nodes but with random choices

Aim: to show the effectiveness of "node selecting scheme", 
core idea: accumulated changes in reservoir w.r.t. node local connectivity --> node "inertia"

by Chengbin Hou 2019
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


class DynRWSG_random(object):
     def __init__(self, G_dynamic, restart_prob=0.2, update_threshold=0.1, emb_dim=128, 
                    num_walks=20, walk_length=80, window=10, workers=20, negative=5, seed=2019):
          self.G_dynamic = G_dynamic.copy()  # a series of dynamic graphs
          self.emb_dim = emb_dim   # node emb dimensionarity
          self.update_threshold = update_threshold  # if 0, update emb for all affected nodes
          self.restart_prob = restart_prob  # restart probability for random walks
          self.num_walks = num_walks  # num of walks start from each node
          self.walk_length = walk_length  # walk length for each walk
          self.window = window  # Skip-Gram parameter
          self.workers = workers  # Skip-Gram parameter
          self.negative = negative  # Skip-Gram parameter
          self.seed = seed  # Skip-Gram parameter

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
                    G0 = self.G_dynamic[t]
                    sentences = simulate_walks(nx_graph=G0, num_walks=self.num_walks, walk_length=self.walk_length, restart_prob=None) #restart_prob=None or 0 --> deepwalk
                    sentences = [[str(j) for j in i] for i in sentences]
                    w2v.build_vocab(sentences=sentences, update=False) # init traning, so update False
                    w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter) # follow w2v constructor
               else:      # online adapting --------------------
                    G0 = self.G_dynamic[t-1]  # previous graph
                    G1 = self.G_dynamic[t]    # current graph
                    node_update_list, self.reservoir = most_affected_nodes(graph_t0=G0, graph_t1=G1, update_threshold=self.update_threshold, reservoir_dict=self.reservoir) # note: self.reservoir for accumulated changes
                    sentences = simulate_walks(nx_graph=G1, num_walks=self.num_walks, walk_length=self.walk_length, restart_prob=self.restart_prob, affected_nodes=node_update_list)
                    sentences = [[str(j) for j in i] for i in sentences]
                    w2v.build_vocab(sentences=sentences, update=True) # online update
                    w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter)

               emb_dict = {} # {nodeID: emb_vector, ...}
               for node in self.G_dynamic[t].nodes():
                    emb_dict[node] = w2v.wv[str(node)]
               self.emb_dicts.append(emb_dict)
               t2 = time.time()
               print(f'DynRWSG sampling and traning time: {(t2-t1):.2f}s --> {t+1}/{len(self.G_dynamic)}')
          
          return self.emb_dicts  # To save memory useage, we can delete DynRWSG model after training

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

def most_affected_nodes(graph_t0, graph_t1, update_threshold, reservoir_dict):
     ''' return most affected nodes as a list to be updated
          based on the accumulated changes between dynamic graphs
     '''
     G0 = graph_t0.copy()
     G1 = graph_t1.copy()

     edge_add = edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges())) # one may directly use streaming added edges if possible
     # print('---> edges added length: ', len(edge_add))
     edge_del = edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges())) # one may directly use streaming added edges if possible
     # print('---> edges deleted length: ', len(edge_del))

     node_affected_by_edge_add = unique_nodes_from_edge_set(edge_add) # unique
     node_affected_by_edge_del = unique_nodes_from_edge_set(edge_del) # unique
     node_affected = list(set(node_affected_by_edge_add + node_affected_by_edge_del)) # unique
     # print('---> nodes affected; length: ', len(node_affected))
     
     node_add = [node for node in node_affected_by_edge_add if node not in G0.nodes()]
     # print('---> node added; length: ', len(node_add))
     # print('---> node added: ', node_add)
     node_del = [node for node in node_affected_by_edge_del if node not in G1.nodes()]
     # print('---> node deleted; length: ', len(node_del))
     # print('---> node deleted: ', node_del)

     # method 1: all affected nodes in G1
     # if len(node_del) > 0: # these nodes are deleted in G1, so no need to update their embeddings
     #     node_update_list = list(set(node_affected) - set(node_del))
          
     # method 2: m most affected nodes in G1 based on (# of affected time on a node)/(its node degree)
     node_update_list = node_add.copy() # newly added (unseen) nodes mush be to update
     exist_node_affected = list(set(node_affected) - set(node_add) - set(node_del))  # affected nodes are in both G0 and G1
     for node in exist_node_affected:
          nbrs_set1 = set(nx.neighbors(G=G1, n=node))
          nbrs_set0 = set(nx.neighbors(G=G0, n=node))
          changes = len( nbrs_set1.union(nbrs_set0) - nbrs_set1.intersection(nbrs_set0) )
          if node in reservoir_dict.keys():
               reservoir_dict[node] += changes # accumulated changes
          else:
               reservoir_dict[node] = changes  # newly added changes

          degree = nx.degree(G=G0, nbunch=node) # node inertia; the larger degrees require the larger changes to trigger
          score = reservoir_dict[node] / degree # may be larger than 1 if the changes are too large w.r.t. its degree
          if score > update_threshold: # -------------  del it from reservoir & append it to update list -------------
               node_update_list.append(node)
               del reservoir_dict[node]
     print(f'num of nodes in reservoir with accumulated changes but not updated {len(list(reservoir_dict))}')
     print(f'all newly added nodes must be updated, {len(node_add)} but ..... ')
     print(f'num of nodes deleted {len(node_del)}, affected {len(node_affected)}, to be updated {len(node_update_list)}')
     
     num_random_nodes = len(node_update_list)
     all_nodes = [node for node in G1.nodes() if node not in node_add]
     node_update_list = list(np.random.choice(all_nodes, num_random_nodes-len(node_add), replace=False)) + node_add.copy()
     print('num_random_nodes', num_random_nodes)

     return node_update_list, reservoir_dict


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
