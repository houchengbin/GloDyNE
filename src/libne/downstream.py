"""
downstream tasks: Link Prediction, Graph Reconstraction, Node Classification
each task is a Python Class
by Chengbin Hou @ 2019
"""

import random
import numpy as np
import networkx as nx

# ----------------------------------------------------------------------------------
# ------------------ link prediction task based on AUC score -----------------------
# ----------------------------------------------------------------------------------
from .utils import cosine_similarity, auc_score, edge_s1_minus_s0
from sklearn.linear_model import LogisticRegression
class lpClassifier(object):
    def __init__(self, emb_dict):
        self.embeddings = emb_dict

    # clf here is a binary logistic regression <-- np.concatenate((start_node_emb, end_node_emb))
    def lr_clf_init(self, graph_t0):
        G0 = graph_t0.copy()
        pos_edges_with_label = [list(item+(1,)) for item in nx.edges(G0)]
        neg_edges_with_label = []
        num = len(pos_edges_with_label)
        i = 0
        for non_edge in nx.non_edges(G0):
            neg_edges_with_label.append(list(non_edge+(0,)))
            i += 1
            if i >= num:
                break
        all_edges_with_label = pos_edges_with_label + neg_edges_with_label
        random.seed(2020)
        random.shuffle(all_edges_with_label)
        all_test_edge = [e[:2] for e in all_edges_with_label]
        all_test_label = [e[2] for e in all_edges_with_label]
        test_size = len(all_test_edge)
        all_edge_feature = []
        for i in range(test_size):
            start_node_emb = np.array(
                self.embeddings[all_test_edge[i][0]])
            end_node_emb = np.array(
                self.embeddings[all_test_edge[i][1]])
            all_edge_feature.append(np.concatenate((start_node_emb, end_node_emb)))
        # print(np.shape(all_edge_feature))
        lr_clf_init = LogisticRegression(random_state=0).fit(all_edge_feature, all_test_label)
        return lr_clf_init

    def update_LR_auc(self, X_test, Y_test, LR_prev=None):
        test_size = len(X_test)
        all_edge_feature = []
        for i in range(test_size):
            start_node_emb = np.array(
                self.embeddings[X_test[i][0]])
            end_node_emb = np.array(
                self.embeddings[X_test[i][1]])
            all_edge_feature.append(np.concatenate((start_node_emb, end_node_emb)))
            
        lr_clf = LR_prev
        if len(Y_test) == 0:
            print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
            auc = 1.0
        else:
            Y_probs = lr_clf.predict_proba(all_edge_feature)[:,1]  # predict; the second col gives prob of true
            auc = auc_score(y_true=Y_test, y_score=Y_probs)
            lr_clf.fit(all_edge_feature, Y_test)  # update model parameters
        print("concat; auc=", "{:.9f}".format(auc))
        return lr_clf
    
    # clf here is a binary logistic regression <-- (start_node_emb-end_node_emb)
    def lr_clf_init_2(self, graph_t0):
        G0 = graph_t0.copy()
        pos_edges_with_label = [list(item+(1,)) for item in nx.edges(G0)]
        neg_edges_with_label = []
        num = len(pos_edges_with_label)
        i = 0
        for non_edge in nx.non_edges(G0):
            neg_edges_with_label.append(list(non_edge+(0,)))
            i += 1
            if i >= num:
                break
        all_edges_with_label = pos_edges_with_label + neg_edges_with_label
        random.seed(2020)
        random.shuffle(all_edges_with_label)
        all_test_edge = [e[:2] for e in all_edges_with_label]
        all_test_label = [e[2] for e in all_edges_with_label]
        test_size = len(all_test_edge)
        all_edge_feature = []
        for i in range(test_size):
            start_node_emb = np.array(
                self.embeddings[all_test_edge[i][0]])
            end_node_emb = np.array(
                self.embeddings[all_test_edge[i][1]])
            all_edge_feature.append(start_node_emb - end_node_emb)
        # print(np.shape(all_edge_feature))
        lr_clf_init = LogisticRegression(random_state=0).fit(all_edge_feature, all_test_label)
        return lr_clf_init

    def update_LR_auc_2(self, X_test, Y_test, LR_prev=None):
        test_size = len(X_test)
        all_edge_feature = []
        for i in range(test_size):
            start_node_emb = np.array(
                self.embeddings[X_test[i][0]])
            end_node_emb = np.array(
                self.embeddings[X_test[i][1]])
            all_edge_feature.append(start_node_emb - end_node_emb)
            
        lr_clf = LR_prev
        if len(Y_test) == 0:
            print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
            auc = 1.0
        else:
            Y_probs = lr_clf.predict_proba(all_edge_feature)[:,1]  # predict; the second col gives prob of true
            auc = auc_score(y_true=Y_test, y_score=Y_probs)
            lr_clf.fit(all_edge_feature, Y_test)  # update model parameters
        print("abs diff; auc=", "{:.9f}".format(auc))
        return lr_clf

    # clf here is simply a similarity/distance metric <-- <start_node_emb, end_node_emb>
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
            print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
            auc = 1.0
        else:
            auc = auc_score(y_true=Y_true, y_score=Y_probs)
        print("cos sim; auc=", "{:.9f}".format(auc))

def gen_test_edge_wrt_changes(graph_t0, graph_t1, seed=None):
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
    
    neg_edges_with_label = [list(item+(0,)) for item in edge_del]
    pos_edges_with_label = [list(item+(1,)) for item in edge_add]

    random.seed(seed)
    all_nodes = list(G0.nodes())

    if len(edge_add) > len(edge_del):
        num = len(edge_add) - len(edge_del)
        start_nodes = np.random.choice(all_nodes, num, replace=True)
        i = 0
        for start_node in start_nodes:
            try:
                non_nbrs = list(nx.non_neighbors(G0, start_node))
                non_nbr = random.sample(non_nbrs, 1).pop()
                non_edge = (start_node, non_nbr)
                if non_edge not in edge_del:
                    neg_edges_with_label.append(list(non_edge+(0,)))
                    i += 1
                if i >= num:
                    break
            except:
                print('Found a fully connected node: ', start_node, 'Ignore it...')
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
    print('---- len(pos_edges_with_label), len(neg_edges_with_label)', len(pos_edges_with_label), len(neg_edges_with_label))
    return pos_edges_with_label, neg_edges_with_label

"""
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
"""
"""
def gen_test_edge_wrt_changes_plus_others(graph_t0, graph_t1, percentage=1.0):
    ''' generate additional the number of (0.1*nodes) of pos_edges and neg_edges respectively for testing
    '''
    G0 = graph_t0.copy() 
    G1 = graph_t1.copy() # use copy to avoid problem caused by G1.remove_node(node)
    pos_edges_with_label, neg_edges_with_label = gen_test_edge_wrt_changes(G0, G1)

    unseen_nodes = set(G1.nodes()) - set(G0.nodes())
    for node in unseen_nodes: # to avoid unseen nodes while testing
        G1.remove_node(node)
    num_other_edges = int(G1.number_of_nodes() * percentage) # if percentage=1.0
                                                             # it is likely to generate one pos_link and one neg_link per node

    exist_neg_edges = [i[:-1] for i in neg_edges_with_label]  #remove label
    i = 0
    for edge in nx.non_edges(G1):
        if edge not in exist_neg_edges:
            neg_edges_with_label.append(list(edge+(0,)))
            i += 1
        if i >= num_other_edges:
            break

    exist_pos_edges = [i[:-1] for i in pos_edges_with_label] # remove label
    i = 0
    for edge in nx.edges(G1):
        if edge not in exist_pos_edges:
            pos_edges_with_label.append(list(edge+(1,)))
            i += 1
        if i >= num_other_edges:
            break
    print('---- len(pos_edges_with_label), len(neg_edges_with_label)', len(pos_edges_with_label), len(neg_edges_with_label))
    return pos_edges_with_label, neg_edges_with_label
"""



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

    def split_train_evaluate(self, X, Y, train_precent, seed=None):
        np.random.seed(seed=seed)
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






# --------------------------------------------------------------------------------------
# ------------- graph reconstruction task based on precision@k score  ====fast version, but require large ROM====
# --------------------------------------------------------------------------------------
from .utils import pairwise_similarity, ranking_precision_score, average_precision_score
class grClassifier(object):
    def __init__(self, emb_dict, rc_graph):
        self.graph = rc_graph
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
        n = len(score_mat)
        score_mat[range(n), range(n)] = 0.0  # set diagonal to 0 -> do not consider itself as the nearest neighbor (node without self loop)
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
            if len(node_list) == 0: # if there is no testing data (dyn networks not changed), set auc to 1
                print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
                pk_list = 1.00
            else:
                node_idx = node_id2idx(self.graph, node_list)
                new_adj_mat = [self.adj_mat[i] for i in node_idx]
                new_score_mat = [self.score_mat[i] for i in node_idx]
                size = len(new_adj_mat)
                for i in range(size):
                    pk_list.append(ranking_precision_score(new_adj_mat[i], new_score_mat[i], k=top_k)) # ranking_precision_score only on node_list
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
            if len(node_list) == 0: # if there is no testing data (dyn networks not changed), set auc to 1
                print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
                pk_list = 1.00
            else:
                node_idx = node_id2idx(self.graph, node_list)
                new_adj_mat = [self.adj_mat[i] for i in node_idx]
                new_score_mat = [self.score_mat[i] for i in node_idx]
                size = len(new_adj_mat)
                for i in range(size):
                    pk_list.append(average_precision_score(new_adj_mat[i], new_score_mat[i], k=top_k)) # average_precision_score only on node_list
        print("average_precision_score=", "{:.9f}".format(np.mean(pk_list)))

def node_id2idx(graph, node_id):
    G = graph
    all_nodes = list(G.nodes())
    node_idx = []
    for node in node_id:
        node_idx.append(all_nodes.index(node))
    return node_idx

def gen_test_node_wrt_changes(graph_t0, graph_t1):
    ''' reconstruct for nodes with any changes
    '''
    from .utils import unique_nodes_from_edge_set
    G0 = graph_t0.copy() 
    G1 = graph_t1.copy() # use copy to avoid problem caused by G1.remove_node(node)
    edge_add = edge_s1_minus_s0(s1=set(G1.edges()), s0=set(G0.edges()))
    edge_del = edge_s1_minus_s0(s1=set(G0.edges()), s0=set(G1.edges()))

    node_affected_by_edge_add = unique_nodes_from_edge_set(edge_add) # unique
    node_affected_by_edge_del = unique_nodes_from_edge_set(edge_del) # unique
    node_affected = list(set(node_affected_by_edge_add + node_affected_by_edge_del)) # unique

    unseen_nodes = list( set(G1.nodes()) - set(G0.nodes()) ) # unique
    test_nodes = [node for node in node_affected if node not in unseen_nodes] # remove unseen nodes
    print('---- len(test_nodes)', len(test_nodes))
    return test_nodes

def gen_test_node_wrt_changes_plus_others(graph_t0, graph_t1):
    ''' currently, we reconstruct whole graph; but todo...
    '''
    pass

"""
# --------------------------------------------------------------------------------------
# ------------- graph reconstruction task based on precision@k score ==== memory saving, batch version, but slow ====
# --------------------------------------------------------------------------------------
from .utils import pairwise_similarity, ranking_precision_score, average_precision_score
class grClassifier_batch(object):
    def __init__(self, emb_dict, rc_graph):
        self.embeddings = emb_dict
        self.graph = rc_graph

    def evaluate_precision_k(self, top_k, node_list=None):
        ''' Mean Precision at rank k
        '''
        G = self.graph.copy()
        adj_mat = nx.to_numpy_array(G=G, nodelist=G.nodes()) # ordered by G.nodes(); n-by-n
        adj_mat = np.where(adj_mat==0, 0, 1) # vectorized implementation weighted -> unweighted if necessary; if memory error, try sparse matrix
        emb_mat = [self.embeddings[node] for node in G.nodes()]

        pk_list = []
        num_nodes = len(G.nodes())
        norm_vec = np.sqrt(np.einsum('ij,ij->i', emb_mat, emb_mat)) # norm of each row of emb_mat  # https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix
        if node_list == None: # eval all nodes
            for i in range(num_nodes): # due to memory issue, we have to eval them one by one...
                score = [np.inner(emb_mat[i],emb_mat[j])/(norm_vec[i]*norm_vec[j]) for j in range(num_nodes)] # cos of i w.r.t each j
                truth = adj_mat[i]
                pk_list.append(ranking_precision_score(y_true=truth, y_score=score, k=top_k))
        else: # only eval on node_list
            if len(node_list) == 0: # if there is no testing data (dyn networks not changed), set auc to 1
                print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
                pk_list = 1.00
            else:
                node_idx = node_id2idx(G, node_list)
                for i in node_idx: # only eval on node_list
                    score = [np.inner(emb_mat[i],emb_mat[j])/(norm_vec[i]*norm_vec[j]) for j in range(num_nodes)] # cos of i w.r.t each j
                    truth = adj_mat[i]
                    pk_list.append(ranking_precision_score(y_true=truth, y_score=score, k=top_k))
        print("ranking_precision_score=", "{:.9f}".format(np.mean(pk_list)))

    def evaluate_average_precision_k(self, top_k, node_list=None):
        ''' Mean Average Precision at rank k
        '''
        G = self.graph.copy()
        adj_mat = nx.to_numpy_array(G=G, nodelist=G.nodes()) # ordered by G.nodes(); n-by-n
        adj_mat = np.where(adj_mat==0, 0, 1) # vectorized implementation weighted -> unweighted if necessary; if memory error, try sparse matrix
        emb_mat = [self.embeddings[node] for node in G.nodes()]

        pk_list = []
        num_nodes = len(G.nodes())
        norm_vec = np.sqrt(np.einsum('ij,ij->i', emb_mat, emb_mat)) # norm of each row of emb_mat  # https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix
        if node_list == None: # eval all nodes
            for i in range(num_nodes): # due to memory issue, we have to eval them one by one...
                score = [np.inner(emb_mat[i],emb_mat[j])/(norm_vec[i]*norm_vec[j]) for j in range(num_nodes)] # cos of i w.r.t each j
                truth = adj_mat[i]
                pk_list.append(average_precision_score(y_true=truth, y_score=score, k=top_k))
        else: # only eval on node_list
            if len(node_list) == 0: # if there is no testing data (dyn networks not changed), set auc to 1
                print('------- NOTE: two graphs do not have any change -> no testing data -> set result to 1......')
                pk_list = 1.00
            else:
                node_idx = node_id2idx(G, node_list)
                for i in node_idx: # only eval on node_list
                    score = [np.inner(emb_mat[i],emb_mat[j])/(norm_vec[i]*norm_vec[j]) for j in range(num_nodes)] # cos of i w.r.t each j
                    truth = adj_mat[i]
                    pk_list.append(average_precision_score(y_true=truth, y_score=score, k=top_k))
        print("average_precision_score=", "{:.9f}".format(np.mean(pk_list)))
"""








"""
# -----------------------------------------------------------------------------------
# ------------- top-k retrive task based on similarity score ------------------------
# -----------------------------------------------------------------------------------
# todo...


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
