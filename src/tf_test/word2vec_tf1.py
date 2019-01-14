'''
A simple implementation of word2vec

Re-implement word2vec using tensorflow_v1.0, according to tf tutorial:
https://www.tensorflow.org/tutorials/representation/word2vec

by Chengbin Hou 2019
'''

import tensorflow as tf

# ------ tf variables
# a random matrix initialized by uniform dist.
# where input parameters -> embeddings i.e. each row is a word embedding
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# output parameters initialized by normal dist.
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# ------ tf placeholders
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) # why 1 column?

# ------ tf embedding lookup
# train_input can take out corresponding embedding 
# by using tf build-in embedding lookup
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# ------ tf model
# Compute the NCE loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nce_weights,
                 biases=nce_biases,
                 labels=train_labels,
                 inputs=embed,
                 num_sampled=num_sampled,
                 num_classes=vocabulary_size))

# ------ tf optimizer
# we use the SGD optimizer for the NCE loss 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)


# ------ tf training



