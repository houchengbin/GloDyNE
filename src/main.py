'''
GloDyNE (or previous called DynWalks)
STEP1: prepare data
STEP2: learn node embeddings
STEP3: evaluate downstream tasks

GloDyNE hyper-parameters:
limit=0.1                          # limited computational resources i.e. the upper limit # of selected nodes
                                   # NOTE: limit i.e. $\alpha$ in our paper
num_walks=10, walk_length=80,      # random walk hyper-parameters
window=10, negative=5,             # Skip-Gram hyper-parameters
seed=2019, workers=32,             # others

--------------------------------------------------------------------------------------
NB: You may ignore other static network embedding methods: DeepWalk, GraRep, HOPE.
    Our method DynWalks is independent from them.

    For other compared dynamic network embedding methods, please see:
    BCGD (2016):        https://github.com/linhongseba/Temporal-Network-Embedding
    DynGEM (2017):      https://github.com/palash1992/DynamicGEM
    DynTriad (2018):    https://github.com/luckiezhou/DynamicTriad
    tNodeEmbed (2019):  https://github.com/urielsinger/tNodeEmbed
--------------------------------------------------------------------------------------

by Chengbin Hou @ 2019
'''

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import time
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import networkx as nx
from libne.utils import load_any_obj_pkl, save_any_obj_pkl


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    # -----------------------------------------------general settings--------------------------------------------------
    parser.add_argument('--graph', default='data/cora/cora_dyn_graphs.pkl',
                        help='graph/network')
    parser.add_argument('--label', default='data/cora/cora_node_label_dict.pkl',
                        help='node label')
    parser.add_argument('--emb-dim', default=128, type=int,
                        help='node embeddings dimensions')
    parser.add_argument('--task', default='all', choices=['lp', 'gr', 'nc', 'all', 'save'],
                        help='choices of downstream tasks: lp, gr, nc, all, save')
    parser.add_argument('--emb-file', default='output/cora_DynWalks_128_embs.pkl',
                        help='node embeddings; suggest: data_method_dim_embs.pkl')
    # -------------------------------------------------method settings-----------------------------------------------------------
    parser.add_argument('--method', default='DynWalks', choices=['DynWalks', 'DeepWalk', 'GraRep', 'HOPE'],
                        help='choices of Network Embedding methods')
    parser.add_argument('--limit', default=0.1, type=float,
                        help='the limit of nodes to be updated at each time step i.e. $\alpha$ in our paper')
    parser.add_argument('--scheme', default=4, type=int,
                        help='1 for greedy, 2 for random, 3 for modularity based, 4 for METIS based')
    # walk based methods
    parser.add_argument('--num-walks', default=10, type=int,
                        help='# of random walks of each node')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='length of each random walk')
    # gensim word2vec parameters
    parser.add_argument('--window', default=10, type=int,
                        help='window size of SGNS model')
    parser.add_argument('--negative', default=5, type=int,
                        help='negative samples of SGNS model')
    parser.add_argument('--workers', default=32, type=int,
                        help='# of parallel processes.')
    parser.add_argument('--seed', default=2019, type=int,
                        help='random seed to fix testing data')
    args = parser.parse_args()
    return args


def main(args):
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print(f'Summary of all settings: {args}')

    # ----------------------------------------STEP1: prepare data-------------------------------------------------------
    print('\nSTEP1: start loading data......')
    t1 = time.time()
    G_dynamic = load_any_obj_pkl(args.graph)
    t2 = time.time()
    print(f'STEP1: end loading data; time cost: {(t2-t1):.2f}s')

    # -----------------------------------STEP2: upstream embedding task-------------------------------------------------
    print('\nSTEP2: start learning embeddings......')
    print(f'The model used: {args.method} -------------------- \
            \nThe # of dynamic graphs: {len(G_dynamic)}; \
            \nThe # of nodes @t_init: {nx.number_of_nodes(G_dynamic[0])}, and @t_last {nx.number_of_nodes(G_dynamic[-1])} \
            \nThe # of edges @t_init: {nx.number_of_edges(G_dynamic[0])}, and @t_last {nx.number_of_edges(G_dynamic[-1])}')
    t1 = time.time()
    model = None
    if args.method == 'DynWalks':
        from libne import DynWalks  
        model = DynWalks.DynWalks(G_dynamic=G_dynamic, limit=args.limit, emb_dim=args.emb_dim, scheme=args.scheme,
                                    num_walks=args.num_walks, walk_length=args.walk_length, window=args.window, negative=args.negative,
                                    workers=args.workers, seed=args.seed)
        model.sampling_traning()
    elif args.method == 'DeepWalk':
        from libne import DeepWalk 
        model = DeepWalk.DeepWalk(G_dynamic=G_dynamic, num_walks=args.num_walks, walk_length=args.walk_length, window=args.window,
                                    negative=args.negative, emb_dim=args.emb_dim, workers=args.workers, seed=args.seed)
        model.sampling_traning()
    else:
        print('method not found...')
        exit(0)
    t2 = time.time()
    print(f'STEP3: end learning embeddings; time cost: {(t2-t1):.2f}s')

    # ---------------------------------------STEP3: downstream task-----------------------------------------------
    print('\nSTEP3: start evaluating ......: ')
    t1 = time.time()
    emb_dicts = model.emb_dicts
    if args.task == 'save':
        save_any_obj_pkl(obj=emb_dicts, path=args.emb_file)
        print(f'Save node embeddings in file: {args.emb_file}')
        print(f'No downsateam task; exit... ')
    del model  # to save memory

    print(f'--- start link prediction task --> use current emb @t to predict **future** changed links @t+1 ...: ')
    if args.task == 'lp_changed' or args.task == 'all':   # the size of LP testing data depends on the changes between two consecutive snapshots
        from libne.downstream import lpClassifier, gen_test_edge_wrt_changes
        for t in range(len(G_dynamic)-1):
            print(f'Current time step @t: {t}')
            print(f'Changed Link Prediction task by AUC score')
            pos_edges_with_label, neg_edges_with_label = gen_test_edge_wrt_changes(G_dynamic[t],G_dynamic[t+1],seed=args.seed)  # diff between edge_set(Gt+1) and edges_set(Gt)
            test_edges = [e[:2] for e in pos_edges_with_label] + [e[:2] for e in neg_edges_with_label]
            test_label = [e[2] for e in pos_edges_with_label] + [e[2] for e in neg_edges_with_label]
            ds_task = lpClassifier(emb_dict=emb_dicts[t])     
            ds_task.evaluate_auc(test_edges, test_label)
    
    print(f'--- start node classification task --> use current emb @t to infer **current* corresponding label @t...: ')
    if args.task == 'nc' or args.task == 'all':
        from libne.downstream import ncClassifier
        from sklearn.linear_model import LogisticRegression
        try:
            label_dict = load_any_obj_pkl(args.label) # ground truth label .pkl
            for t in range(len(G_dynamic)-1):         # ignore the last one... so that it is consistent with LP
                print(f'Current time step @t: {t}')
                X = []
                Y = []
                for node in G_dynamic[t].nodes():     # only select current available nodes for eval, testing all nodes in Gt
                    X.append(node)
                    Y.append(str(label_dict[node]))   # label as str, otherwise, sklearn error
                ds_task = ncClassifier(emb_dict=emb_dicts[t], clf=LogisticRegression())   
                ds_task.split_train_evaluate(X, Y, train_precent=0.5, seed=args.seed)
        except:
            print('no node label; no NC task')

    print(f'--- start graph/link reconstraction task --> use current emb @t to reconstruct **current** graph @t...: ')
    if args.task == 'gr' or args.task == 'all':
        from libne.downstream import grClassifier, gen_test_node_wrt_changes
        for t in range(len(G_dynamic)-1):             # ignore the last one... so that it is consistent with LP
            print(f'Current time step @t: {t}') 
            ds_task = grClassifier(emb_dict=emb_dicts[t], rc_graph=G_dynamic[t])
            changed_nodes = gen_test_node_wrt_changes(G_dynamic[t],G_dynamic[t+1])                     # CGR testing nodes, changed nodes between Gt and Gt+1
            print('# of changed_nodes for testing: ', len(changed_nodes))  
            all_nodes = list(G_dynamic[t].nodes())
            np.random.seed(seed = args.seed)
            random_nodes = list(np.random.choice(all_nodes, int(len(all_nodes)*1.0), replace=False))   # GR testing nodes, 20% of all nodes
            print('# of random_nodes for testing: ', len(random_nodes))                                 
            # ------------------------- @5 ----------------------
            precision_at_k = 5
            #print(f'Changed Graph Reconstruction by AP @{precision_at_k}')
            #ds_task.evaluate_precision_k(top_k=precision_at_k, node_list=changed_nodes)             # CGR AP
            print(f'Graph Reconstruction by AP @{precision_at_k}')
            ds_task.evaluate_precision_k(top_k=precision_at_k, node_list=random_nodes)              # GR AP
            #print(f'Changed Graph Reconstruction by MAP @{precision_at_k}')
            #ds_task.evaluate_average_precision_k(top_k=precision_at_k, node_list=changed_nodes)     # CGR MAP
            #print(f'Graph Reconstruction by MAP @{precision_at_k}')
            #ds_task.evaluate_average_precision_k(top_k=precision_at_k, node_list=random_nodes)      # GR MAP
            # ------------------------- @10 ---------------------
            precision_at_k = 10
            #print(f'Changed Graph Reconstruction by AP @{precision_at_k}')
            #ds_task.evaluate_precision_k(top_k=precision_at_k, node_list=changed_nodes)             # CGR AP
            print(f'Graph Reconstruction by AP @{precision_at_k}')
            ds_task.evaluate_precision_k(top_k=precision_at_k, node_list=random_nodes)              # GR AP
            #print(f'Changed Graph Reconstruction by MAP @{precision_at_k}')
            #ds_task.evaluate_average_precision_k(top_k=precision_at_k, node_list=changed_nodes)     # CGR MAP
            #print(f'Graph Reconstruction by MAP @{precision_at_k}')
            #ds_task.evaluate_average_precision_k(top_k=precision_at_k, node_list=random_nodes)      # GR MAP
            # ------------------------- @20 ---------------------
            precision_at_k = 20
            #print(f'Changed Graph Reconstruction by AP @{precision_at_k}')
            #ds_task.evaluate_precision_k(top_k=precision_at_k, node_list=changed_nodes)             # CGR AP
            print(f'Graph Reconstruction by AP @{precision_at_k}')
            ds_task.evaluate_precision_k(top_k=precision_at_k, node_list=random_nodes)              # GR AP
            #print(f'Changed Graph Reconstruction by MAP @{precision_at_k}')
            #ds_task.evaluate_average_precision_k(top_k=precision_at_k, node_list=changed_nodes)     # CGR MAP
            #print(f'Graph Reconstruction by MAP @{precision_at_k}')
            #ds_task.evaluate_average_precision_k(top_k=precision_at_k, node_list=random_nodes)      # GR MAP
            # ------------------------- @40 ---------------------
            precision_at_k = 40
            #print(f'Changed Graph Reconstruction by AP @{precision_at_k}')
            #ds_task.evaluate_precision_k(top_k=precision_at_k, node_list=changed_nodes)             # CGR AP
            print(f'Graph Reconstruction by AP @{precision_at_k}')
            ds_task.evaluate_precision_k(top_k=precision_at_k, node_list=random_nodes)              # GR AP
            #print(f'Changed Graph Reconstruction by MAP @{precision_at_k}')
            #ds_task.evaluate_average_precision_k(top_k=precision_at_k, node_list=changed_nodes)     # CGR MAP
            #print(f'Graph Reconstruction by MAP @{precision_at_k}')
            #ds_task.evaluate_average_precision_k(top_k=precision_at_k, node_list=random_nodes)      # GR MAP
            # NOTE: if memory error, try grClassifier_batch (see dowmstream.py) which is slow but greatly reduce ROM
    
    t2 = time.time()
    print(f'STEP3: end evaluating; time cost: {(t2-t1):.2f}s')


if __name__ == '__main__':
    print(f'------ START @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')
    main(parse_args())
    print(f'------ END @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')
    