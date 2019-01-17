'''
Dynamic Word2Vec example to address the following questions
by Chengbin Hou

# basic useage of Gemsim Word2Vec: given sentences -> train embedding [done]

# how to update for unseen words [done]
# how to continue traning on seen + unseen words [done]
# check if updated embedding and similarity are changed? [yes]
# how Gensim Word2Vec update embedding [done]

# how to reset input embedding matrix [done]
# can I take out output embedding matrix? and reset? [yes]

# how to update embedding using one pair --> train_sg_pair [to do...]
# check if more freq a pair occurs, more closer they are [yes]
# check if and only if the center node are updated, but its positive sample and negative samples are remained [to do...]
OR try to directly use sentences? walks? but how to obtain good walks???

# how to use lock factor [to do...]
# try in networks [to do...]

# sklearn pca 2d vis [done]
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


# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['a','b'],['b','c'],['c','d']]

w2v = gensim.models.Word2Vec(size=6, window=1, sg=1, hs=0, negative=1, ns_exponent=1.0,
                            alpha=0.025, min_alpha=0.0001, min_count=1, sample=0.001, iter=5, workers=4, seed=2019,
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


