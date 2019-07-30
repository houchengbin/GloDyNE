'''
other static NE method...
DeepWalk-SGNS version by Chengbin Hou
'''

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import time
import random
import pickle

import gensim
import logging
import numpy as np
import networkx as nx


class DeepWalk(object):
     def __init__(self, G_dynamic, emb_dim=128, num_walks=20, walk_length=80, window=10, 
                    workers=20, negative=5, seed=2019, restart_prob=None):
          self.G_dynamic = G_dynamic.copy()  # a series of dynamic graphs
          self.emb_dim = emb_dim   # node emb dimensionarity
          self.restart_prob = restart_prob  # restart probability for random walks; if None --> DeepWalk
          self.num_walks = num_walks  # num of walks start from each node
          self.walk_length = walk_length  # walk length for each walk
          self.window = window  # Skip-Gram parameter
          self.workers = workers  # Skip-Gram parameter
          self.negative = negative  # Skip-Gram parameter
          self.seed = seed  # Skip-Gram parameter

          self.emb_dicts = [] # emb_dict @ t0, t1, ...; len(self.emb_dicts) == len(self.G_dynamic)
          
     def sampling_traning(self):     
          for t in range(len(self.G_dynamic)):
               t1 = time.time()
               G0 = self.G_dynamic[t]
               w2v = gensim.models.Word2Vec(sentences=None, size=self.emb_dim, window=self.window, sg=1, hs=0, negative=self.negative, ns_exponent=0.75,
                                   alpha=0.025, min_alpha=0.0001, min_count=1, sample=0.001, iter=4, workers=self.workers, seed=self.seed,
                                   corpus_file=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                                   max_vocab_size=None, max_final_vocab=None, trim_rule=None)  # w2v constructor, default parameters
               sentences = simulate_walks(nx_graph=G0, num_walks=self.num_walks, walk_length=self.walk_length, restart_prob=None) #restart_prob=None or 0 --> deepwalk
               sentences = [[str(j) for j in i] for i in sentences]
               w2v.build_vocab(sentences=sentences, update=False) # init traning, so update False
               w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter) # follow w2v constructor

               emb_dict = {} # {nodeID: emb_vector, ...}
               for node in self.G_dynamic[t].nodes():
                    emb_dict[node] = w2v.wv[str(node)]
               self.emb_dicts.append(emb_dict)
               t2 = time.time()
               print(f'DeepWalk sampling and traning time: {(t2-t1):.2f}s --> {t+1}/{len(self.G_dynamic)} graphs')
          
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

def simulate_walks(nx_graph, num_walks, walk_length, restart_prob=None, affected_nodes=None):
     '''
     Repeatedly simulate random walks from each node
     '''
     G = nx_graph
     walks = []

     if affected_nodes == None: # simulate walks on every node in the graph [offline] --> deepwalk
          nodes = list(G.nodes())
     else:                     # simulate walks on affected nodes [online]
          nodes = list(affected_nodes)
     
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
