'''
Dynamic RW-SGNS demo for Link Prediction task
by Chengbin HOU & Han ZHANG

todo:
1) reset deledted nodes or continue traning
2) what about the deleted nodes while doing negative sampling
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
from utils import gen_test_edge_wrt_changes, load_dynamic_graphs, edge_s1_minus_s0, unique_nodes_from_edge_set
from downstream import lpClassifier

def most_affected_nodes(graph_t0, graph_t1, reservoir_dict=None, update_threshold=0.2):
     G0 = graph_t0.copy()
     G1 = graph_t1.copy()

     edge_add = edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges()))
     # print('---> edges added length: ', len(edge_add))
     edge_del = edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges()))
     # print('---> edges deleted length: ', len(edge_del))

     node_affected_by_edge_add = unique_nodes_from_edge_set(edge_add) # unique
     node_affected_by_edge_del = unique_nodes_from_edge_set(edge_del) # unique
     node_affected = list(set(node_affected_by_edge_add + node_affected_by_edge_del)) # unique
     print('---> nodes affected; length: ', len(node_affected))
     
     node_add = [node for node in node_affected_by_edge_add if node not in G0.nodes()]
     print('---> node added; length: ', len(node_add))
     # print('---> node added: ', node_add)
     node_del = [node for node in node_affected_by_edge_del if node not in G1.nodes()]
     print('---> node deleted; length: ', len(node_del))
     # print('---> node deleted: ', node_del)

     # method 1: all affected nodes in G1
     # if len(node_del) > 0: # these nodes are deleted in G1, so no need to update their embeddings
     #     node_update_list = list(set(node_affected) - set(node_del))
          
     # method 2: m most affected nodes in G1 based on (# of affected time on a node)/(its node degree)
     node_update_list = node_add # newly added (unseen) nodes mush be to update
     exist_node_affected = list(set(node_affected) - set(node_del) - set(node_add))  # affected nodes are in both G0 and G1
     for node in exist_node_affected:
          nbrs_set1 = set(nx.neighbors(G=G1, n=node))
          nbrs_set0 = set(nx.neighbors(G=G0, n=node))
          changes = len( nbrs_set1.union(nbrs_set0) - nbrs_set1.intersection(nbrs_set0) )
          if node in reservoir_dict.keys():
               reservoir_dict[node] += changes # accumulated changes
          else:
               reservoir_dict[node] = changes  # newly added changes

          degree = nx.degree(G=G0, nbunch=node) # node inertia; the larger degree the more likely not updated
          score = reservoir_dict[node] / degree # may be larger than 1 if the changes are too large w.r.t. its degree
          if score > update_threshold: # -------------  del it from reservoir & append it to update list -------------
               node_update_list.append(node)
               del reservoir_dict[node]

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
               sentences = simulate_walks(nx_graph=G0, num_walks=20, walk_length=80, restart_prob=None)
               sentences = [[str(j) for j in i] for i in sentences]
               # print('sentences[:10]', sentences[:10]) # if set restart_prob=1, each sentence only contains itself
               w2v.build_vocab(sentences=sentences, update=False) # init traning, so update False
               w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter) # follow w2v constructor
               
               emb_dict = {} # nodeID: emb_vector
               for node in G_dynamic[t].nodes():
                    emb_dict[node] = w2v.wv[str(node)]

               if t < len(G_dynamic)-1:
                    print('Link Prediction task, time step @: ', t)
                    pos_edges_with_label, neg_edges_with_label = gen_test_edge_wrt_changes(G_dynamic[t],G_dynamic[t+1]) #take this out!!!!!!!!!!!!!!!!!!!!!!!!!
                    test_edges = [e[:2] for e in pos_edges_with_label] + [e[:2] for e in neg_edges_with_label]
                    test_label = [e[2] for e in pos_edges_with_label] + [e[2] for e in neg_edges_with_label]
                    ds_task = lpClassifier(emb_dict=emb_dict)  # similarity/distance metric as clf; basically, lp is a binary clf probelm
                    ds_task.evaluate_auc(test_edges, test_label)

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
          
          reservoir = {} # {nodeID: # of times affected, ...}
          for t in range(len(G_dynamic)):
               t3 = time.time()
               if t ==0:
                    G0 = G_dynamic[t]
                    sentences = simulate_walks(nx_graph=G0, num_walks=20, walk_length=80, restart_prob=None)
                    sentences = [[str(j) for j in i] for i in sentences]
                    # print('sentences[:10]', sentences[:10]) # if set restart_prob=1, each sentence only contains itself
                    w2v.build_vocab(sentences=sentences, update=False) # init traning, so update False
                    w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter) # follow w2v constructor
               else:
                    G0 = G_dynamic[t-1]
                    G1 = G_dynamic[t]
                    node_update_list, reservoir = most_affected_nodes(graph_t0=G0, graph_t1=G1, reservoir_dict=reservoir, update_threshold=0.2)
                    sentences = simulate_walks(nx_graph=G1, num_walks=20, walk_length=80, restart_prob=None, affected_nodes=node_update_list)
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
                    print('Link Prediction task, time step @: ', t)
                    pos_edges_with_label, neg_edges_with_label = gen_test_edge_wrt_changes(G_dynamic[t],G_dynamic[t+1])
                    test_edges = [e[:2] for e in pos_edges_with_label] + [e[:2] for e in neg_edges_with_label]
                    test_label = [e[2] for e in pos_edges_with_label] + [e[2] for e in neg_edges_with_label]
                    ds_task = lpClassifier(emb_dict=emb_dict)  # similarity/distance metric as clf; basically, lp is a binary clf probelm
                    ds_task.evaluate_auc(test_edges, test_label)

               t4 = time.time()
               print(f'current time step; time cost: {(t4-t3):.2f}s')
          t2 = time.time()
          print(f'Dynamic NE -> all time steps; time cost: {(t2-t1):.2f}s')