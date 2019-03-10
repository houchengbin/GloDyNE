"""
downstream tasks; each task is a class;
by Chengbin Hou & Zeyu Dong
"""

import random
import numpy as np
import networkx as nx


# ----------------------------------------------------------------------------------
# ------------- node classification task based on F1 score -------------------------
# ----------------------------------------------------------------------------------
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
class ncClassifier(object):
    def __init__(self, emb_dict, clf):
        self.embeddings = emb_dict
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



# ----------------------------------------------------------------------------------
# ------------------ link prediction task based on AUC score -----------------------
# ----------------------------------------------------------------------------------
from .utils import cosine_similarity, auc_score, edge_s1_minus_s0
class lpClassifier(object):
    def __init__(self, emb_dict):
        self.embeddings = emb_dict

    # clf here is simply a similarity/distance metric
    def evaluate_auc(self, X_test, Y_test):
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
        if len(Y_true) == 0: # if there is no testing data (dyn networks not changed), set auc to 1
            print('------- NOTE: two graphs do not have any change -> no testing data -> set auc to 1......')
            auc = 1.0
        else:
            auc = auc_score(y_true=Y_true, y_score=Y_probs)
        print("auc=", "{:.9f}".format(auc))

def gen_test_edge_wrt_changes(graph_t0, graph_t1):
    ''' input: two networkx graphs
        generate **changed** testing edges for link prediction task
        currently, we only consider pos_neg_ratio = 1.0
        return: pos_edges_with_label [(node1, node2, 1), (), ...]
                neg_edges_with_label [(node3, node4, 0), (), ...]
    '''
    G0 = graph_t0.copy() 
    G1 = graph_t1.copy() # use copy to avoid problem caused by G1.remove_node(node)
    edge_add = edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges()))
    edge_del = edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges()))
    unseen_nodes = set(G1.nodes()) - set(G0.nodes())
    for node in unseen_nodes: # to avoid unseen nodes while testing
        G1.remove_node(node)
    
    edge_add_unseen_node = [] # to avoid unseen nodes while testing
    #print('len(edge_add)', len(edge_add))
    for node in unseen_nodes: 
        for edge in edge_add:
            if node in edge:
                edge_add_unseen_node.append(edge)
    edge_add = edge_add - set(edge_add_unseen_node)
    #print('len(edge_add)', len(edge_add))
    pos_edges_with_label = [list(item+(1,)) for item in edge_add]
    neg_edges_with_label = [list(item+(0,)) for item in edge_del]
    if len(edge_add) > len(edge_del):
        num = len(edge_add) - len(edge_del)
        i = 0
        for non_edge in nx.non_edges(G1):
            if non_edge not in edge_del:
                neg_edges_with_label.append(list(non_edge+(0,)))
                i += 1
            if i >= num:
                break
    elif len(edge_add) < len(edge_del):
        num = len(edge_del) - len(edge_add)
        i = 0
        for edge in nx.edges(G1):
            if edge not in edge_add:
                pos_edges_with_label.append(list(edge+(1,)))
                i += 1
            if i >= num:
                break
    else: # len(edge_add) == len(edge_del)
        pass
    return pos_edges_with_label, neg_edges_with_label

# --------------------------------------------------------------------------------------
# ------------- graph reconstruction task based on precision@k score -------------------
# --------------------------------------------------------------------------------------
from .utils import pairwise_similarity, ranking_precision_score, average_precision_score
class grClassifier(object):
    def __init__(self, emb_dict, rc_graph):
        self.embeddings = emb_dict
        self.adj_mat, self.score_mat = self.gen_test_data_wrt_graph_truth(graph=rc_graph)
    
    def gen_test_data_wrt_graph_truth(self, graph):
        ''' input: a networkx graph
            output: adj matrix and score matrix; note both matrices are symmetric
        '''
        G = graph.copy()
        adj_mat = nx.to_numpy_array(G=G, nodelist=None) # ordered by G.nodes(); n-by-n
        adj_mat = np.where(adj_mat==0, 0, 1) # vectorized implementation weighted -> unweighted if necessary
        emb_mat = []
        for node in G.nodes():
            emb_mat.append(self.embeddings[node])
        score_mat = pairwise_similarity(emb_mat, type='cosine') # n-by-n corresponding to adj_mat
        return np.array(adj_mat), np.array(score_mat)

    def evaluate_precision_k(self, top_k, node_list=None):
        ''' Precision at rank k; to be merged with average_precision_score()
        '''
        pk_list = []
        if node_list==None: # eval all nodes
            size = self.adj_mat.shape[0] # num of rows -> num of nodes
            for i in range(size):
                pk_list.append(ranking_precision_score(self.adj_mat[i], self.score_mat[i], k=top_k)) # ranking_precision_score
        else: # only eval on node_list
            print('currently not support specified node list')
            exit(0)
        print("ranking_precision_score=", "{:.9f}".format(np.mean(pk_list)))

    def evaluate_average_precision_k(self, top_k, node_list=None):
        ''' Average precision at rank k; to be merged with evaluate_precision_k()
        '''
        pk_list = []
        if node_list==None: # eval all nodes
            size = self.adj_mat.shape[0] # num of rows -> num of nodes
            for i in range(size):
                pk_list.append(average_precision_score(self.adj_mat[i], self.score_mat[i], k=top_k)) # average_precision_score
        else: # only eval on node_list
            print('currently not support specified node list')
            exit(0)
        print("average_precision_score=", "{:.9f}".format(np.mean(pk_list)))


# -----------------------------------------------------------------------------------
# ------------- top-k retrive task based on similarity score ------------------------
# -----------------------------------------------------------------------------------
# todo...



"""
# --------------------------------------------------------------------------------
# ------------------------- 2D/3d visualization task -----------------------------
# --------------------------------------------------------------------------------
from sklearn.decomposition import PCA
from matplotlib import pyplot
def pca_vis(model):
    ''' simple vis use matplotlib
    input: word2vec model [change to model.vv using pickle]
    output: vis
    '''
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


def tf_vis():
    ''' vis using tensorflow
    see vis.py file
    to do...
    ''' 
    pass
"""