'''
Dynamic RW-SGNS demo for Link Prediction task
by Chengbin HOU & Han ZHANG
'''

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
import logging
import time
import random
import pickle
import numpy as np
import networkx as nx
from utils import load_dynamic_graphs, edge_s1_minus_s0, unique_nodes_from_edge_set
from downstream import grClassifier

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

     if restart_prob == None: # naive random walk
          for walk_iter in range(num_walks):
               t1 = time.time()
               random.shuffle(nodes)
               for node in nodes:
                    walks.append(random_walk(nx_graph=G, start_node=node, walk_length=walk_length))
               t2 = time.time()
               # print(f'Walk iteration: {walk_iter+1}/{num_walks}; time cost: {(t2-t1):.2f}')
     else: # random walk with restart
          for walk_iter in range(num_walks):
               t1 = time.time()
               random.shuffle(nodes)
               for node in nodes:
                    walks.append(random_walk_restart(nx_graph=G, start_node=node, walk_length=walk_length, restart_prob=restart_prob))
               t2 = time.time()
               # print(f'Walk iteration: {walk_iter+1}/{num_walks}; time cost: {(t2-t1):.2f}')
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



if __name__ == '__main__':
     # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
     # community_dict = load_any_obj(path='../data/synthetic_LFR/LFR_community_dict.data') # ground truth
     G_dynamic = load_dynamic_graphs('../data/AS733/AS733_dyn_graphs.pkl')

     is_dyn = True
     if not is_dyn:
          # ------ DeepWalk
          t1 = time.time()

          # SGNS and suggested parameters to be tuned: size, window, negative, workers, seed
          # to tune other parameters, please read https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec

          for t in range(len(G_dynamic)):                    
               t3 = time.time()
               G0 = G_dynamic[t]
               w2v = gensim.models.Word2Vec(sentences=None, size=128, window=10, sg=1, hs=0, negative=5, ns_exponent=0.75,
                    alpha=0.025, min_alpha=0.0001, min_count=1, sample=0.001, iter=4, workers=8, seed=2019,
                    corpus_file=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                    max_vocab_size=None, max_final_vocab=None, trim_rule=None) # w2v constructor
               sentences = simulate_walks(nx_graph=G0, num_walks=10, walk_length=80, restart_prob=None)
               sentences = [[str(j) for j in i] for i in sentences]
               # print('sentences[:10]', sentences[:10]) # if set restart_prob=1, each sentence only contains itself
               w2v.build_vocab(sentences=sentences, update=False) # init traning, so update False
               w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter) # follow w2v constructor
               
               emb_dict = {} # nodeID: emb_vector
               for node in G_dynamic[t].nodes():
                    emb_dict[node] = w2v.wv[str(node)]

               if t < len(G_dynamic)-1:
                    print('Graph Reconstruction task, time step @: ', t)
                    ds_task = grClassifier(emb_dict=emb_dict, rc_graph=G_dynamic[t])  # reconstruct current graph @ t
                    ds_task.evaluate_precision_k(top_k=20)
                    ds_task.evaluate_average_precision_k(top_k=20)

               t4 = time.time()
               print(f'current time step; time cost: {(t4-t3):.2f}s')
          t2 = time.time()
          print(f'Static NE -> all time steps; time cost: {(t2-t1):.2f}s')

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
     else:
          # ------ DynSGNE
          t1 = time.time()

          # SGNS and suggested parameters to be tuned: size, window, negative, workers, seed
          # to tune other parameters, please read https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
          w2v = gensim.models.Word2Vec(sentences=None, size=128, window=10, sg=1, hs=0, negative=5, ns_exponent=0.75,
                              alpha=0.025, min_alpha=0.0001, min_count=1, sample=0.001, iter=4, workers=8, seed=2019,
                              corpus_file=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                              max_vocab_size=None, max_final_vocab=None, trim_rule=None) # w2v constructor

          for t in range(len(G_dynamic)):
               t3 = time.time()
               if t ==0:
                    G0 = G_dynamic[t]
                    sentences = simulate_walks(nx_graph=G0, num_walks=10, walk_length=80, restart_prob=None)
                    sentences = [[str(j) for j in i] for i in sentences]
                    # print('sentences[:10]', sentences[:10]) # if set restart_prob=1, each sentence only contains itself
                    w2v.build_vocab(sentences=sentences, update=False) # init traning, so update False
                    w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter) # follow w2v constructor
               else:
                    G0 = G_dynamic[t-1]
                    G1 = G_dynamic[t]
                    edge_add = edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges()))
                    # print('---> edges added length: ', len(edge_add))
                    edge_del = edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges()))
                    # print('---> edges deleted length: ', len(edge_del))

                    node_affected_by_edge_add = unique_nodes_from_edge_set(edge_add)
                    node_affected_by_edge_del = unique_nodes_from_edge_set(edge_del)
                    node_affected = list(set(node_affected_by_edge_add + node_affected_by_edge_del))
                    print('---> nodes affected; length: ', len(node_affected))
                    
                    node_add = [node for node in node_affected_by_edge_add if node not in G0.nodes()]
                    # print('---> node added; length: ', len(node_add))
                    print('---> node added: ', node_add)
                    node_del = [node for node in node_affected_by_edge_del if node not in G1.nodes()]
                    # print('---> node deleted; length: ', len(node_del))
                    print('---> node deleted: ', node_del)
                    if len(node_del) > 0: # these nodes are deleted in G1, so no need to update their embeddings
                         node_affected = list(set(node_affected) - set(node_del))

                    sentences = simulate_walks(nx_graph=G1, num_walks=10, walk_length=80, restart_prob=None, affected_nodes=node_affected)
                    sentences = [[str(j) for j in i] for i in sentences]
                    # print('sentences[:10] updated', sentences[:10]) # if set restart_prob=1, each sentence only contains itself

                    w2v.build_vocab(sentences=sentences, update=True) # online update
                    w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter)

               # print('\n -----------------------------------------------')
               # print('wv for node 0 ', w2v.wv['0'])
               # print('similar_by_word for node 0', w2v.wv.similar_by_word('0'))
               # pca_vis(w2v)

               emb_dict = {} # nodeID: emb_vector
               for node in G_dynamic[t].nodes():
                    emb_dict[node] = w2v.wv[str(node)]
               '''
               # save emb as a dict
               time_step = str(t)
               path = '../output/emb_t' + time_step  
               save_emb(emb_dict=emb_dict, path=path)
               # lemb_dict = load_emb(path)
               '''
               # generate equal numbers of positive and negative edges for LP test
               if t < len(G_dynamic)-1:
                    print('Graph Reconstruction task, time step @: ', t)
                    ds_task = grClassifier(emb_dict=emb_dict, rc_graph=G_dynamic[t])  # reconstruct current graph @ t
                    ds_task.evaluate_precision_k(top_k=1)
                    ds_task.evaluate_precision_k(top_k=2)
                    ds_task.evaluate_precision_k(top_k=3)
                    ds_task.evaluate_precision_k(top_k=4)
                    ds_task.evaluate_precision_k(top_k=5)
                    ds_task.evaluate_average_precision_k(top_k=5)

               t4 = time.time()
               print(f'current time step; time cost: {(t4-t3):.2f}s')
          t2 = time.time()
          print(f'Dynamic NE -> all time steps; time cost: {(t2-t1):.2f}s')