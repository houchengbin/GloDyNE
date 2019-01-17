'''
Dynamic RW-SGNS demo
by Chengbin Hou

# 1) generate two dynamic graphs
# 2) random walks on the most affected  -> RW or RWR or RW-BFS-DFS -> walks
     and retrain related nodes based on new walks, 
     while keeping unrelated nodes (not in walks) unchanged
# 4) by now, we have embeddings for two dynamic graphs, we may do the following task:
     4.1) anomaly detection: one node in A community goes to B community
     4.2) node classification: how label changes? but label change should related to structure?
          based on G1 -> Z2; G2-G1 -> Z2 and do node classification
          or similar to 4.1) ......

     4.3) link prediction: 
          based on Z1; G1 -> Z1 -> ask if G2-G1 added edges or G1-G2 deleted edges
          or similar to 4.4) ...... based on Z2; G1 -> Z1; G2-G1 -> Z2 and do link prediction
     4.4) k most similar nodes: paper citation network.... retrive paper titles for a paper 
          based on G1->Z1, G2-G1 -> Z2 and retrive on Z2 w.r.t. offline on Z2
'''

import gensim
import logging


def pca_vis(model):
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


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['a','b','c','d','e','f','g'],['a','b','c','d','e','f','g']]

w2v = gensim.models.Word2Vec(size=6, window=1, sg=1, hs=0, negative=1, ns_exponent=0.0,
                            alpha=0.025, min_alpha=0.0001, min_count=1, sample=0.001, iter=4, workers=4, seed=2019,
                            sentences=None, corpus_file=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                            max_vocab_size=None, max_final_vocab=None, trim_rule=None)

w2v.build_vocab(sentences=sentences, update=False)
w2v.train(sentences=sentences,total_examples=w2v.corpus_count, epochs=w2v.iter) #change iter??? #reset input embedding matrix to 0

print('\n -----------------------------------------------')
print('wv for a ', w2v.wv['a'])
print('similar_by_word for ', w2v.wv.similar_by_word('a'))
pca_vis(w2v)

''' dump and load --> no affect
import pickle
myw2v = pickle.dumps(w2v)
del w2v
w2v = pickle.loads(myw2v)
'''

# w2v.wv.vectors[0] = [1,2,3,4,5,6] #reset input embedding
# w2v.syn1neg[0] = [7,8,9,10,11,12] #reset output embedding
sentences = [['d','e'],['e','f']]
w2v.build_vocab(sentences=sentences, update=True) #Copy all the existing weights, and reset the weights to zeros for the newly added vocabulary
w2v.train(sentences=sentences,total_examples=w2v.corpus_count, epochs=w2v.iter)
print('\n -----------------------------------------------')
print('wv for a ', w2v.wv['a']) # b, c, d
print('similar_by_word for a', w2v.wv.similar_by_word('a'))
pca_vis(w2v)



sentences = [['a','b'],['b','a'],['a','f'],['f','a']]
w2v.build_vocab(sentences=sentences, update=True) #
w2v.train(sentences=sentences,total_examples=w2v.corpus_count, epochs=w2v.iter)
print('\n -----------------------------------------------')
print('wv for a ', w2v.wv['a'])
print('similar_by_word for a', w2v.wv.similar_by_word('a'))
pca_vis(w2v)


