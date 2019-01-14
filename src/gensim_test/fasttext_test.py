'''
test 

# any2vec https://radimrehurek.com/gensim/models/base_any2vec.html#gensim.models.base_any2vec.BaseWordEmbeddingsModel
# smooth: https://blog.csdn.net/baimafujinji/article/details/51297802
# n-gram: https://zhuanlan.zhihu.com/p/32829048

'''

from gensim.models import KeyedVectors # https://radimrehurek.com/gensim/models/keyedvectors.html
from gensim.models import FastText # https://radimrehurek.com/gensim/models/fasttext.html
from gensim.models import Word2Vec # https://radimrehurek.com/gensim/models/word2vec.html


from gensim.models import FastText

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

model = FastText(sentences, min_count=1)
say_vector = model['say']  # get vector for word
print(say_vector)
of_vector = model['of']  # get vector for out-of-vocab word
print(of_vector)