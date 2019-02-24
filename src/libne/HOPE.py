import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from sklearn.preprocessing import normalize
import time

__author__ = "Alan WANG"
__email__ = "alan1995wang@outlook.com"


class HOPE(object):
    def __init__(self, G_dynamic, emb_dim=128):
        self._d = emb_dim
        self.G_dynamic = G_dynamic.copy()
        self.emb_dicts = [] # emb_dict @ t0, t1, ...; len(self.emb_dicts) == len(self.G_dynamic)

    def traning(self):
        for t in range(len(self.G_dynamic)):
            t1 = time.time()
            G0 = self.G_dynamic[t]
            A = nx.to_numpy_matrix(G0)

            # self._beta = 0.0728

            # M_g = np.eye(graph.number_of_nodes()) - self._beta * A
            # M_l = self._beta * A

            M_g = np.eye(G0.number_of_nodes())
            M_l = np.dot(A, A)

            S = np.dot(np.linalg.inv(M_g), M_l)
            # s: \sigma_k
            u, s, vt = lg.svds(S, k=self._d // 2)
            sigma = np.diagflat(np.sqrt(s))
            X1 = np.dot(u, sigma)
            X2 = np.dot(vt.T, sigma)
            # self._X = X2
            self._X = np.concatenate((X1, X2), axis=1)

            emb_dict = {} # {nodeID: emb_vector, ...}
            look_back = list(self.G_dynamic[t].nodes())
            for i, embedding in enumerate(self._X):
                emb_dict[look_back[i]] = embedding
            self.emb_dicts.append(emb_dict)
            t2 = time.time()
            print(f'HOPE traning time: {(t2-t1):.2f}s --> {t+1}/{len(self.G_dynamic)} graphs')
          
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