"""
demo of evaluating node embedding for different downstream task(s)

Note that, the 'main.py' file contains all functions in 'eval.py'. You may test DynWalks in one go using 'main.py' itself by set task=all.

For fairness, this 'eval.py' as well as all files in 'libne' will be used to evalate other dynamic network embedding methods
To generate node embeddings by other methods, please see:
    BCGD (2016):        https://github.com/linhongseba/Temporal-Network-Embedding
    DynGEM (2017):      https://github.com/palash1992/DynamicGEM
    DynLINE (2018):     https://github.com/lundu28/DynamicNetworkEmbedding
    DynTriad (2018):    https://github.com/luckiezhou/DynamicTriad
    tNodeEmbed (2019):  https://github.com/urielsinger/tNodeEmbed
    GloDyNE (2020):     https://github.com/houchengbin/GloDyNE
You may need to convert node embeddings in the same format as GloDyNE does. Please refer to sampling_traning() in DynWalks.py.
"""

import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from libne.utils import load_any_obj_pkl

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--graph', default='data/cora/cora_dyn_graphs.pkl',
                        help='graph/network file')
    parser.add_argument('--label', default='data/cora/cora_node_label_dict.pkl',
                        help='node label file')
    parser.add_argument('--emb-file', default='output/cora_DynWalks_128_embs.pkl',
                        help='node embeddings file; suggest: data_method_dim_embs.pkl')
    parser.add_argument('--task', default='all', choices=['lp', 'gr','lp_changed', 'gr_changed', ' nc', 'gr', 'all', 'save'],
                        help='choices of downstream tasks: lp, nc, gr, all, save')
    parser.add_argument('--seed', default=2019, type=int,
                        help='random seed')                     # also used to fix the random testing data
    args = parser.parse_args()
    return args

def main(args):
    print(f'Summary of all settings: {args}')

    # ---------------------------------------STEP1: prepare data-----------------------------------------------------
    print('\nSTEP1: start loading data......')
    t1 = time.time()
    G_dynamic = load_any_obj_pkl(args.graph)
    emb_dicts = load_any_obj_pkl(args.emb_file)
    t2 = time.time()
    print(f'STEP1: end loading data; time cost: {(t2-t1):.2f}s')
    
    # ---------------------------------------STEP3: downstream task-----------------------------------------------
    print('\nSTEP3: start evaluating ......: ')
    t1 = time.time()
            
    # Link Prediction (LP)
    print(f'--- start link prediction task --> use current emb @t to predict **future** changed links @t+1 ...: ')
    if args.task == 'lp_changed' or args.task == 'all':   # the size of LP testing data depends on the changes between two consecutive snapshots
        from libne.downstream import lpClassifier, gen_test_edge_wrt_changes
        for t in range(len(G_dynamic)-1):
            print(f'Current time step @t: {t}')
            print(f'Changed Link Prediction task (via cos sim) by AUC score')
            pos_edges_with_label, neg_edges_with_label = gen_test_edge_wrt_changes(G_dynamic[t],G_dynamic[t+1],seed=args.seed)  # diff between edge_set(Gt+1) and edges_set(Gt)
            test_edges = [e[:2] for e in pos_edges_with_label] + [e[:2] for e in neg_edges_with_label]
            test_label = [e[2] for e in pos_edges_with_label] + [e[2] for e in neg_edges_with_label]
            ds_task = lpClassifier(emb_dict=emb_dicts[t])     
            ds_task.evaluate_auc(test_edges, test_label)
    
    # Node Classification (NC)
    print(f'--- start node classification task 0.5 --> use current emb @t to infer **current* corresponding label @t...: ')
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
    print(f'--- start node classification task 0.7 --> use current emb @t to infer **current* corresponding label @t...: ')
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
                ds_task.split_train_evaluate(X, Y, train_precent=0.7, seed=args.seed)
        except:
            print('no node label; no NC task')
    print(f'--- start node classification task 0.9 --> use current emb @t to infer **current* corresponding label @t...: ')
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
                ds_task.split_train_evaluate(X, Y, train_precent=0.9, seed=args.seed)
        except:
            print('no node label; no NC task')

    # Graph Reconstraction (GR)
    print(f'--- start graph/link reconstraction task --> use current emb @t to reconstruct **current** graph @t: ')
    if args.task == 'gr' or args.task == 'all':
        from libne.downstream import grClassifier, gen_test_node_wrt_changes
        for t in range(len(G_dynamic)-1): # ignore the last one, so that the length consistent with Changed LP
            print(f'Current time step @t: {t}')
            ds_task = grClassifier(emb_dict=emb_dicts[t], rc_graph=G_dynamic[t])
            node_list = None  # GR testing nodes
            #node_list = gen_test_node_wrt_changes(G_dynamic[t],G_dynamic[t+1]) # changed nodes between Gt and Gt+1
            #print('# of changed_nodes for testing: ', len(changed_nodes))
            #all_nodes = list(G_dynamic[t].nodes())
            #node_list = list(np.random.choice(all_nodes, int(len(all_nodes)*0.2), replace=False)) # 20% random nodes
            if node_list == None:
                print('node_list == None --> test for all nodes in a graph')
            else:
                print('# of node_list for testing: ', len(node_list))
            # --- @1 ---
            k = 1
            print(f'Graph Reconstruction by P @{k}')
            ds_task.evaluate_precision_k(top_k=k, node_list=node_list)            # GR P@k or MeanP@k over node_list
            #print(f'Graph Reconstruction by AP @{k}')
            #ds_task.evaluate_average_precision_k(top_k=k, node_list=node_list)   # GR AP@k or MeanAP@k (a.k.a. MAP@k in many papers) over node_list
            # --- @5 ---
            k = 5
            print(f'Graph Reconstruction by P @{k}')
            ds_task.evaluate_precision_k(top_k=k, node_list=node_list)
            # --- @10 ---
            k = 10
            print(f'Graph Reconstruction by P @{k}')
            ds_task.evaluate_precision_k(top_k=k, node_list=node_list)
            # --- @20 ---
            k = 20
            print(f'Graph Reconstruction by P @{k}')
            ds_task.evaluate_precision_k(top_k=k, node_list=node_list)
            # --- @40 ---
            k = 40
            print(f'Graph Reconstruction by P @{k}')
            ds_task.evaluate_precision_k(top_k=k, node_list=node_list)
            # --- @100 ---
            k = 100
            print(f'Graph Reconstruction by P @{k}')
            ds_task.evaluate_precision_k(top_k=k, node_list=node_list)
            # --- @1000 ---
            # Although we may be more interested in small P@K, as most networks have small average degree,
            # it is also intesting to see the algorithm performance for P@1000 (hopefully ~ 100%).
            #k = 1000
            #print(f'Graph Reconstruction by P @{k}')
            #ds_task.evaluate_precision_k(top_k=k, node_list=node_list)
            #print(f'Graph Reconstruction by AP @{k}')
            #ds_task.evaluate_average_precision_k(top_k=k, node_list=node_list)
            # If OOM, try grClassifier_batch (see dowmstream.py) which is slow but requires much smaller memory 
    t2 = time.time()
    print(f'STEP3: end evaluating; time cost: {(t2-t1):.2f}s')


if __name__ == '__main__':
    print(f'------ START @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')
    main(parse_args())
    print(f'------ END @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')