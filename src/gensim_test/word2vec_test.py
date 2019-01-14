'''
learn gensim word2vec
https://radimrehurek.com/gensim/models/word2vec.html

re-organized by Chengbin HOU with comments
'''


# ====== basic examples ======

# --- a simple example
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
path1 = "word2vec.model"
model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
model.save(path1)
''' another example
from gensim.models import Word2Vec
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(min_count=1)
model.build_vocab(sentences)  # prepare the model vocabulary
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)  # train word vectors
'''

# --- load model and continue training but not use online
model = Word2Vec.load(path1)
model.train([["hello", "world"]], total_examples=1, epochs=1)
model.save(path1)

# --- take out word vectors if exist
vector1 = model.wv["computer"]
# print(vector)
# vector = model.wv['hello'] # not exist due to not online
# vector = model.wv['chengbin']

# --- KeyedVectors
from gensim.models import KeyedVectors
path2 = "wordvectors.kv"    # much smaller than path1
model.wv.save(path2)
wv = KeyedVectors.load(path2, mmap='r')
vector2 = wv['computer']  # numpy vector of a word
print("vector1 == vector2 \n", vector1 == vector2)

''' not able to continuous training on KeyedVectors
It is impossible to continue training the vectors loaded from the C format 
because the hidden weights, vocabulary frequencies and the binary tree are missing. 
To continue training, you’ll need the full Word2Vec object state, as stored by save(), 
not just the KeyedVectors.
'''

# --- another way to get KeyedVectors & del model & pickle dump load
word_vectors = model.wv
del model
import pickle
# pickle in memory
dumps_word_vectors = pickle.dumps(word_vectors)
loads_word_vectors = pickle.loads(dumps_word_vectors)
print("vector1 == loads_word_vectors['computer'] \n", vector1 == loads_word_vectors['computer'])
# pickle in disk
path3 = "vectors"
with open(path3, 'wb') as f:
    pickle.dump(word_vectors, f)
with open(path3, 'rb') as f:
    load_word_vectors = pickle.load(f)
    vector3 = load_word_vectors['computer']  # numpy vector of a word
    print("vector1 == vector3 \n", vector1 == vector3)







# ====== basic APIs under class gensim.models.word2vec ======

# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
'''
parameters:
class gensim.models.word2vec.Word2Vec(sentences=None, corpus_file=None, size=100, alpha=0.025, 
window=5, min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, 
negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=<built-in function hash>, iter=5, null_word=0, trim_rule=None, 
sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=(), max_final_vocab=None)

methods and attributes:
wv, vocabulary, trainables, load(), save(), train()
accuracy(), build_vocab(), build_vocab_from_freq(), clear_sims(), cum_table, delete_temporary_training_data(),
doesnt_match(), doesnt_match(), evaluate_word_pairs(), get_latest_training_loss(), hashfxn, init_sims(), 
intersect_word2vec_format(), iter, layer1_size, min_count, most_similar(), most_similar_cosmul(), n_similarity(),
predict_output_word(), reset_from(), sample, save_word2vec_format(), score(), similar_by_vector(), similar_by_word(),
similarity(), syn0_lockf, syn1, syn1neg, wmdistance()
'''


# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2VecTrainables
'''
parameters:
class gensim.models.word2vec.Word2VecTrainables(vector_size=100, seed=1, hashfxn=<built-in function hash>)

methods and attributes:
load(), save(), prepare_weights(), reset_weights(), seeded_vector(), update_weights()
'''



# ====== advanced APIs under class gensim.models.word2vec ======
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2VecVocab

# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.score_cbow_pair
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.score_sg_pair
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.train_cbow_pair
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.train_sg_pair

# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.BrownCorpus
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.LineSentence
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.PathLineSentences
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Text8Corpus
