"""
a matrix factorization based NE method: GraRep
originally from https://github.com/thunlp/OpenNE/blob/master/src/openne/grarep.py
"""

import numpy as np
from numpy import linalg as la
from sklearn.preprocessing import normalize
import networkx as nx
import time
from scipy import sparse

class GraRep(object):
    def __init__(self, G_dynamic, emb_dim=128, Kstep=4):
        self.G_dynamic = G_dynamic.copy()  # a series of dynamic graphs
        self.Kstep = Kstep
        assert emb_dim % Kstep == 0
        self.dim = int(emb_dim/Kstep)

        self.emb_dicts = [] # emb_dict @ t0, t1, ...; len(self.emb_dicts) == len(self.G_dynamic)

    def getAdjMat(self, graph):
        adj = get_adj_mat(graph)  # for isolated node row, normalize to [1/n, 1/n, ...]
        return row_as_probdist(adj, dense_output=True, preserve_zeros=False)

    def GetProbTranMat(self, Ak):
        probTranMat = np.log(Ak/np.tile(
            np.sum(Ak, axis=0), (self.node_size, 1))) \
            - np.log(1.0/self.node_size)
        probTranMat[probTranMat < 0] = 0
        probTranMat[probTranMat == np.nan] = 0
        return probTranMat

    def GetRepUseSVD(self, probTranMat, alpha):
        U, S, VT = la.svd(probTranMat)
        Ud = U[:, 0:self.dim]
        Sd = S[0:self.dim]
        return np.array(Ud)*np.power(Sd, alpha).reshape((self.dim))

    def traning(self):
        for t in range(len(self.G_dynamic)):
            t1 = time.time()
            G0 = self.G_dynamic[t]
            self.adj = self.getAdjMat(G0)
            self.node_size = self.adj.shape[0]
            self.Ak = np.matrix(np.identity(self.node_size))
            self.RepMat = np.zeros((self.node_size, int(self.dim*self.Kstep)))
            for i in range(self.Kstep):
                print('Kstep =', i)
                self.Ak = np.dot(self.Ak, self.adj)
                probTranMat = self.GetProbTranMat(self.Ak)
                Rk = self.GetRepUseSVD(probTranMat, 0.5)
                Rk = normalize(Rk, axis=1, norm='l2')
                self.RepMat[:, self.dim*i:self.dim*(i+1)] = Rk[:, :]

            emb_dict = {} # {nodeID: emb_vector, ...}
            look_back = list(self.G_dynamic[t].nodes())
            for i, embedding in enumerate(self.RepMat):
                emb_dict[look_back[i]] = embedding
            self.emb_dicts.append(emb_dict)
            t2 = time.time()
            print(f'GraRep traning time: {(t2-t1):.2f}s --> {t+1}/{len(self.G_dynamic)} graphs')
          
        return self.emb_dicts  # To save memory useage, we can delete DynRWSG model after training


    def save_emb(self, path='unnamed_dyn_emb_dicts.pkl'):
        ''' save # emb_dict @ t0, t1, ... to a file using pickle
        '''
        with open(path, 'wb') as f:
            pickle.dump(self.emb_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_emb(self, path='unnamed_dyn_emb_dicts.pkl'):
        ''' load # emb_dict @ t0, t1, ... to a file using pickle
        '''
        with open(path, 'rb') as f:
            any_object = pickle.load(f)
        return any_object




def row_as_probdist(mat, dense_output=False, preserve_zeros=False):
    """Make each row of matrix sums up to 1.0, i.e., a probability distribution.
    Support both dense and sparse matrix.
    Attributes
    ----------
    mat : scipy sparse matrix or dense matrix or numpy array
        The matrix to be normalized
    dense_output : bool
        whether forced dense output
    perserve_zeros : bool
        If False, for row with all entries 0, we normalize it to a vector with all entries 1/n.
        Leave 0 otherwise
    Returns
    -------
    dense or sparse matrix:
        return dense matrix if input is dense matrix or numpy array
        return sparse matrix for sparse matrix input
        (note: np.array & np.matrix are diff; and may cause some dim issues...)
    """
    row_sum = np.array(mat.sum(axis=1)).ravel()  # type: np.array
    zero_rows = row_sum == 0
    row_sum[zero_rows] = 1
    diag = sparse.dia_matrix((1 / row_sum, 0), (mat.shape[0], mat.shape[0]))
    mat = diag.dot(mat)
    if not preserve_zeros:
        mat += sparse.csr_matrix(zero_rows.astype(int)).T.dot(sparse.csr_matrix(np.repeat(1 / mat.shape[1], mat.shape[1])))

    if dense_output and sparse.issparse(mat):
        return mat.todense()
    return mat

def get_adj_mat(graph, is_sparse=True):
    """ return adjacency matrix; \n
        use 'csr' format for sparse matrix \n
    """
    if is_sparse:
        return nx.to_scipy_sparse_matrix(graph, format='csr', dtype='float64')
    else:
        return nx.to_numpy_matrix(graph, dtype='float64')