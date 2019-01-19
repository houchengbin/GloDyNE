"""
downstream tasks; each task is a class;
by Chengbin Hou & Zeyu Dong
"""

import math
import random

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


# ------------------node classification task---------------------------

class ncClassifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)  # here clf is LR
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = np.random.get_state()
        training_size = int(train_precent * len(X))
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        np.random.set_state(state)
        return self.evaluate(X_test, Y_test)

    def train(self, X, Y, Y_all):
        # to support multi-labels, fit means dict mapping {orig cat: binarized vec}
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        # since we have use Y_all fitted, then we simply transform
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        # see TopKRanker(OneVsRestClassifier)
        # the top k probs to be output...
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def evaluate(self, X, Y):
        # multi-labels, diff len of labels of each node
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)  # pred val of X_test i.e. Y_pred
        Y = self.binarizer.transform(Y)  # true val i.e. Y_test
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        print(results)
        return results
class TopKRanker(OneVsRestClassifier):  # orignal LR or SVM is for binary clf
    def predict(self, X, top_k_list):  # re-define predict func of OneVsRestClassifier
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[
                probs_.argsort()[-k:]].tolist()  # denote labels
            probs_[:] = 0  # reset probs_ to all 0
            probs_[labels] = 1  # reset probs_ to 1 if labels denoted...
            all_labels.append(probs_)
        return np.asarray(all_labels)


# ------------------link prediction task---------------------------
class lpClassifier(object):

    def __init__(self, vectors):
        self.embeddings = vectors

    # clf here is simply a similarity/distance metric
    def evaluate(self, X_test, Y_test, seed=0):
        test_size = len(X_test)
        Y_true = [int(i) for i in Y_test]
        Y_probs = []
        for i in range(test_size):
            start_node_emb = np.array(
                self.embeddings[X_test[i][0]]).reshape(-1, 1)
            end_node_emb = np.array(
                self.embeddings[X_test[i][1]]).reshape(-1, 1)
            # ranging from [-1, +1]
            score = cosine_similarity(start_node_emb, end_node_emb)
            # switch to prob... however, we may also directly y_score = score
            Y_probs.append((score + 1) / 2.0)
            # in sklearn roc... which yields the same reasult
        roc = roc_auc_score(y_true=Y_true, y_score=Y_probs)
        if roc < 0.5:
            roc = 1.0 - roc  # since lp is binary clf task, just predict the opposite if<0.5
        print("roc=", "{:.9f}".format(roc))

def norm(a):
    sum = 0.0
    for i in range(len(a)):
        sum = sum + a[i] * a[i]
    return math.sqrt(sum)

def cosine_similarity(a, b):
    sum = 0.0
    for i in range(len(a)):
        sum = sum + a[i] * b[i]
    return sum / (norm(a) * norm(b) + 1e-100)

'''
def lp_train_test_split(graph, ratio=0.8, neg_pos_link_ratio=1.0):
    # randomly split links/edges into training set and testing set
    # *** note: we do not assume every node must be connected after removing links
    # *** hence, the resulting graph might have few single nodes --> more realistic scenario
    # *** e.g. a user just sign in a website has no link to others

    # graph: OpenANE graph data strcture
    # ratio: perc of links for training; ranging [0, 1]
    # neg_pos_link_ratio: 1.0 means neg-links/pos-links = 1.0 i.e. balance case; raning [0, +inf)
    g = graph
    print("links for training {:.2f}%, and links for testing {:.2f}%, neg_pos_link_ratio is {:.2f}".format(
        ratio * 100, (1 - ratio) * 100, neg_pos_link_ratio))
    test_pos_sample = []
    test_neg_sample = []
    train_size = int(ratio * len(g.G.edges))
    test_size = len(g.G.edges) - train_size

    # generate testing set that contains both pos and neg samples
    test_pos_sample = random.sample(g.G.edges(), int(test_size))
    test_neg_sample = []
    num_neg_sample = int(test_size * neg_pos_link_ratio)
    num = 0
    while num < num_neg_sample:
        pair_nodes = np.random.choice(g.look_back_list, size=2, replace=False)
        if pair_nodes not in g.G.edges():
            num += 1
            test_neg_sample.append(list(pair_nodes))

    test_edge_pair = test_pos_sample + test_neg_sample
    test_edge_label = list(np.ones(len(test_pos_sample))) + \
        list(np.zeros(len(test_neg_sample)))

    print('before removing, the # of links: ', g.numDiEdges(),
          ';   the # of single nodes: ', g.numSingleNodes())
    # training set should NOT contain testing set i.e. delete testing pos samples
    g.G.remove_edges_from(test_pos_sample)
    print('after removing,  the # of links: ', g.numDiEdges(),
          ';   the # of single nodes: ', g.numSingleNodes())
    print("# training links {0}; # positive testing links {1}; # negative testing links {2},".format(
        g.numDiEdges(), len(test_pos_sample), len(test_neg_sample)))
    return g.G, test_edge_pair, test_edge_label
'''