'''
Dynamic Word2Vec example
by Chengbin Hou
ref: https://rutumulkar.com/blog/2015/word2vec and https://radimrehurek.com/gensim/models/word2vec.html
'''

import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = gensim.models.word2vec.LineSentence("C:/Users/cheng/Desktop/_code/DynNE/data/text8-files/text8-rest")

w2v = gensim.models.Word2Vec(size=128, window=10, sg=1, hs=0, negative=5, ns_exponent=1.0,
                            alpha=0.025, min_alpha=0.0001, min_count=1, sample=0.001, iter=5, workers=4, seed=2019,
                            sentences=None, corpus_file=None, sorted_vocab=1, batch_words=10000, compute_loss=False,
                            max_vocab_size=None, max_final_vocab=None, trim_rule=None)
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec


w2v.build_vocab(sentences=sentences, update=False)
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.build_vocab
w2v.train(sentences=sentences,total_examples=w2v.corpus_count, epochs=w2v.iter)
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.train
print('w2v.n_similarity(["king"], ["duke"])', w2v.n_similarity(["king"], ["duke"]))
# print('w2v.n_similarity(["king"], ["queen"])', w2v.n_similarity(["king"], ["queen"])) # error OK!

print('emb for king ---v1 ', w2v.wv['king'])
print('emb for duke ---v1 ', w2v.wv['duke'])




sentences2 = gensim.models.word2vec.LineSentence("C:/Users/cheng/Desktop/_code/DynNE/data/text8-files/text8-queen")
w2v.build_vocab(sentences=sentences, update=True)
# print('emb for queen --- inti', w2v.wv['queen'])

w2v.train(sentences=sentences,total_examples=w2v.corpus_count, epochs=w2v.iter)
print('w2v.n_similarity(["king"], ["duke"])', w2v.n_similarity(["king"], ["duke"]))
print('w2v.n_similarity(["king"], ["queen"])', w2v.n_similarity(["king"], ["queen"])) # now OK!

print('emb for king --- v2', w2v.wv['king'])
print('emb for duke --- v2', w2v.wv['duke'])
print('emb for queen --- v2', w2v.wv['queen'])



'''

# https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.Word2VecKeyedVectors
'''


'''

# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2VecVocab
'''


'''
# Represents the inner shallow neural network used to train Word2Vec.
# class gensim.models.word2vec.Word2VecTrainables(vector_size=100, seed=1, hashfxn=<built-in function hash>)
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2VecTrainables
'''

# https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.most_similar
# https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.n_similarity
# https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similar_by_vector
# https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similar_by_word
# predict_output_word(context_words_list, topn=10)
# reset_from(other_model)
# https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.similarity

# https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.Doc2VecKeyedVectors.save_word2vec_format


