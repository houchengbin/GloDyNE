'''
Dynamic RW-SGNS demo
by Chengbin Hou

# 1) generate two dynamic graphs
# 2) random walks on the most affected  -> RW or RWR or RW-BFS-DFS -> walks
     and retrain related nodes based on new walks, 
     while keeping unrelated nodes (not in walks) unchanged
# 3) by now, we have embeddings for two dynamic graphs, we may do the following task:
     3.1) anomaly detection: one node in A community goes to B community
     3.2) node classification: how label changes? but label change should related to structure?
          based on G1 -> Z2; G2-G1 -> Z2 and do node classification
          or similar to 4.1) ......

     3.3) link prediction: 
          based on Z1; G1 -> Z1 -> ask if G2-G1 added edges or G1-G2 deleted edges
          or similar to 4.4) ...... based on Z2; G1 -> Z1; G2-G1 -> Z2 and do link prediction
     3.4) k most similar nodes: paper citation network.... retrive paper titles for a paper 
          based on G1->Z1, G2-G1 -> Z2 and retrive on Z2 w.r.t. offline on Z2
'''

import gensim
import logging
import time
import random


def pca_vis(model):
     '''
     input: word2vec model [change to model.vv using pickle]
     output: vis
     '''
     from sklearn.decomposition import PCA
     from matplotlib import pyplot
     # fit a 2d PCA model to the vectors
     X = model[model.wv.vocab]
     pca = PCA(n_components=2,random_state=None)
     result = pca.fit_transform(X)
     # create a scatter plot of the projection
     pyplot.scatter(result[:, 0], result[:, 1])
     words = list(model.wv.vocab)
     for i, word in enumerate(words):
          pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
     pyplot.show()

def load_dynamic_graphs(path):
     '''
     return a series of networkx dynamic graphs
     '''
     import pickle
     with open(path, 'rb') as f:
          dynamic_graphs = pickle.load(f)
     return dynamic_graphs

def load_static_graph(path):
     '''
     return a static networkx graph
     '''
     import pickle
     with open(path, 'rb') as f:
          static_graph = pickle.load(f)
     return static_graph

def save_embeddings(path):
     '''
     save embeddings to disk [todo...]
     '''
     pass



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


def edge_s1_minus_s0(s1, s0, is_directed=False):
     if not is_directed:
          s1_reordered = set( (a,b) if a<b else (b,a) for a,b in s1 )
          s0_reordered = set( (a,b) if a<b else (b,a) for a,b in s0 )
          return s1_reordered-s0_reordered
     else:
          print('currently not support directed case')


def unique_nodes_from_edge_set(edge_set):
     unique_nodes = []
     for a, b in edge_set:
          if a not in unique_nodes:
               unique_nodes.append(a)
          if b not in unique_nodes:
               unique_nodes.append(b)
     return unique_nodes


if __name__ == '__main__':
     # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

     '''
     # ------ static NE
     G_static = load_static_graph('../data/synthetic_LFR/LFR_static_graph.data')
     sentences = simulate_walks(nx_graph=G_static, num_walks=20, walk_length=10, restart_prob=None)
     sentences = [[str(j) for j in i] for i in sentences]
     print('sentences[:10]', sentences[:10]) # if set restart_prob=1, each sentence only contains itself

     # SGNS and suggested parameters to be tuned: size, window, negative, workers, seed
     # to tune other parameters, please read https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
     w2v = gensim.models.Word2Vec(size=20, window=5, sg=1, hs=0, negative=3, ns_exponent=0.75,
                         alpha=0.025, min_alpha=0.0001, min_count=1, sample=0.001, iter=4, workers=4, seed=2019,
                         sentences=None, corpus_file=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                         max_vocab_size=None, max_final_vocab=None, trim_rule=None) # w2v constructor

     w2v.build_vocab(sentences=sentences, update=False) # init traning, so update False
     w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter) # follow w2v constructor

     print('\n -----------------------------------------------')
     # print('wv for node 0 ', w2v.wv['0'])
     print('similar_by_word for node 0', w2v.wv.similar_by_word('0'))
     pca_vis(w2v)
     '''

     # ------ dynamic NE
     G_dynamic = load_dynamic_graphs('../data/synthetic_LFR/LFR_dynamic_graphs.data')

     G0 = G_dynamic[0]
     sentences = simulate_walks(nx_graph=G0, num_walks=20, walk_length=10, restart_prob=None)
     sentences = [[str(j) for j in i] for i in sentences]
     print('sentences[:10]', sentences[:10]) # if set restart_prob=1, each sentence only contains itself

     # SGNS and suggested parameters to be tuned: size, window, negative, workers, seed
     # to tune other parameters, please read https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
     w2v = gensim.models.Word2Vec(size=20, window=5, sg=1, hs=0, negative=3, ns_exponent=0.75,
                         alpha=0.025, min_alpha=0.0001, min_count=1, sample=0.001, iter=4, workers=4, seed=2019,
                         sentences=None, corpus_file=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                         max_vocab_size=None, max_final_vocab=None, trim_rule=None) # w2v constructor

     w2v.build_vocab(sentences=sentences, update=False) # init traning, so update False
     w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter) # follow w2v constructor

     print('\n -----------------------------------------------')
     # print('wv for node 0 ', w2v.wv['0'])
     print('similar_by_word for node 0', w2v.wv.similar_by_word('0'))
     # pca_vis(w2v)



     G1 = G_dynamic[1]
     edge_add = edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges()))
     edge_del = edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges()))
     print('---> edges added: ', edge_add)
     print('---> edges deleted: ', edge_del)
     node_affected_by_edge_add = unique_nodes_from_edge_set(edge_add)
     node_affected_by_edge_del = unique_nodes_from_edge_set(edge_del)
     node_affected = node_affected_by_edge_add + node_affected_by_edge_del
     print('---> nodes affected: ', node_affected)

     sentences = simulate_walks(nx_graph=G1, num_walks=20, walk_length=10, restart_prob=None, affected_nodes=node_affected)
     sentences = [[str(j) for j in i] for i in sentences]
     print('sentences[:10] updated', sentences[:10]) # if set restart_prob=1, each sentence only contains itself

     w2v.build_vocab(sentences=sentences, update=True) # online update
     w2v.train(sentences=sentences, total_examples=w2v.corpus_count, epochs=w2v.iter)

     print('\n -----------------------------------------------')
     # print('wv for node 0 ', w2v.wv['0'])
     print('similar_by_word for node 0', w2v.wv.similar_by_word('0'))
     # pca_vis(w2v)
     
