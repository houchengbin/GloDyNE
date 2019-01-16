'''
Dynamic Word2Vec example
by Chengbin Hou
ref: https://rutumulkar.com/blog/2015/word2vec and https://radimrehurek.com/gensim/models/word2vec.html
'''

import gensim
import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = [['a', 'b', 'c'], ['d', 'e', 'f'],['a', 'b', 'c']]

w2v = gensim.models.Word2Vec(size=6, window=1, sg=1, hs=0, negative=1, ns_exponent=1.0,
                            alpha=0.025, min_alpha=0.0001, min_count=1, sample=0.001, iter=5, workers=4, seed=2019,
                            sentences=None, corpus_file=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                            max_vocab_size=None, max_final_vocab=None, trim_rule=None)

w2v.build_vocab(sentences=sentences, update=False) #try True for the first time?
w2v.train(sentences=sentences,total_examples=w2v.corpus_count, epochs=w2v.iter) #change iter???

print('\n -----------------------------------------------')
print('wv for a ', w2v.wv['a'])
print('similar_by_word for ', w2v.wv.similar_by_word('a'))

''' dump and load --> no affect
import pickle
myw2v = pickle.dumps(w2v)
del w2v
w2v = pickle.loads(myw2v)
'''

w2v.wv.vectors[0] = [1,2,3,4,5,6]
sentences = [['a', 'd', 'f'],['a', 'd', 'f'],['a', 'd', 'f'],['a', 'd', 'f'],['a', 'd', 'f'],['a', 'd', 'f'],['a', 'd', 'f']]
w2v.build_vocab(sentences=sentences, update=True)
w2v.train(sentences=sentences,total_examples=w2v.corpus_count, epochs=w2v.iter)
print('\n -----------------------------------------------')
print('wv for a ', w2v.wv['a'])
print('similar_by_word for a', w2v.wv.similar_by_word('a'))
# 试试相似度变化大不大
# 怎么自己来重置某一个向量；或者不充值继续训练？
# 看看他的update那个到底做了什么


sentences = [['d', 'a', 'f'],['d', 'a', 'f'],['d', 'a', 'f']]
w2v.build_vocab(sentences=sentences, update=True)
w2v.train(sentences=sentences,total_examples=w2v.corpus_count, epochs=w2v.iter)
print('\n -----------------------------------------------')
print('wv for a ', w2v.wv['a'])
print('similar_by_word for a', w2v.wv.similar_by_word('a'))
