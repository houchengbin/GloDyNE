'''
demo of evaluating node embedding for different downsatream task(s)

Note that, the 'main.py' file contains all functions in 'eval.py'. You may test DynWalks in one go using 'main.py' itself by set task=all.

For fairness, 
this 'eval.py' as well as all files in 'libne' will be used to evalate other dynamic network embedding methods
To generate node embeddings by other methods, please see:
BCGD:        https://github.com/linhongseba/Temporal-Network-Embedding
DynGEM:      https://github.com/palash1992/DynamicGEM
DynTriad:    https://github.com/luckiezhou/DynamicTriad

by Chengbin Hou
'''

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

    # ---------------------------------------STEP2: downstream task-----------------------------------------------
    print('\nSTEP2: start evaluating ......: ')
    t1 = time.time()
    print(f'--- start link prediction task --> use current emb @t to predict **future** changed links @t+1 ...: ')
    if args.task == 'lp_changed' or args.task == 'all':   # for this task, we may need a big diff between two time steps --> more testing data --> more stable result
        from libne.downstream import lpClassifier, gen_test_edge_wrt_changes
        for t in range(len(emb_dicts)-1):
            print(f'Current time step @t: {t}')
            print(f'Changed Link Prediction task by AUC score')
            pos_edges_with_label, neg_edges_with_label = gen_test_edge_wrt_changes(G_dynamic[t],G_dynamic[t+1]) # use current emb @t predict graph t+1
            test_edges = [e[:2] for e in pos_edges_with_label] + [e[:2] for e in neg_edges_with_label]
            test_label = [e[2] for e in pos_edges_with_label] + [e[2] for e in neg_edges_with_label]
            ds_task = lpClassifier(emb_dict=emb_dicts[t])  # use current emb @t
            ds_task.evaluate_auc(test_edges, test_label)
    
    print(f'--- start graph/link reconstraction task --> use current emb @t to reconstruct **current** graph @t...: ')
    if args.task == 'gr' or args.task == 'all':
        from libne.downstream import grClassifier, gen_test_node_wrt_changes
        for t in range(len(emb_dicts)-1):
            print(f'Current time step @t: {t}')
            ds_task = grClassifier(emb_dict=emb_dicts[t], rc_graph=G_dynamic[t]) # use current emb @t reconstruct graph t
            changed_nodes = gen_test_node_wrt_changes(G_dynamic[t],G_dynamic[t+1])                 # CGR testing nodes
            print('# of changed_nodes for testing: ', len(changed_nodes))  
            all_nodes = list(G_dynamic[t].nodes())
            random_nodes = list(np.random.choice(all_nodes, int(len(all_nodes)*0.25), replace=False))   # GR testing nodes
            print('# of random_nodes for testing: ', len(random_nodes))                                 
            # ------------------------- @10 ----------------------
            precision_at_k = 10
            print(f'Changed Graph Reconstruction by AP @{precision_at_k}')
            ds_task.evaluate_precision_k(top_k=precision_at_k, node_list=changed_nodes)             # CGR AP
            print(f'Graph Reconstruction by AP @{precision_at_k}')
            ds_task.evaluate_precision_k(top_k=precision_at_k, node_list=random_nodes)              # GR AP
            print(f'Changed Graph Reconstruction by MAP @{precision_at_k}')
            ds_task.evaluate_average_precision_k(top_k=precision_at_k, node_list=changed_nodes)     # CGR MAP
            print(f'Graph Reconstruction by MAP @{precision_at_k}')
            ds_task.evaluate_average_precision_k(top_k=precision_at_k, node_list=random_nodes)      # GR AMP
            # ------------------------- @100 ---------------------
            precision_at_k = 100
            print(f'Changed Graph Reconstruction by AP @{precision_at_k}')
            ds_task.evaluate_precision_k(top_k=precision_at_k, node_list=changed_nodes)             # CGR AP
            print(f'Graph Reconstruction by AP @{precision_at_k}')
            ds_task.evaluate_precision_k(top_k=precision_at_k, node_list=random_nodes)              # GR AP
            print(f'Changed Graph Reconstruction by MAP @{precision_at_k}')
            ds_task.evaluate_average_precision_k(top_k=precision_at_k, node_list=changed_nodes)     # CGR MAP
            print(f'Graph Reconstruction by MAP @{precision_at_k}')
            ds_task.evaluate_average_precision_k(top_k=precision_at_k, node_list=random_nodes)      # GR AMP
            # NOTE: if memory error, try grClassifier_batch (see dowmstream.py) which is slow but greatly reduce ROM
    """  
    if args.task == 'lp_changed' or args.task == 'all':
        from libne.downstream import lpClassifier, gen_test_edge_wrt_changes
        for t in range(len(emb_dicts)-1):
            print(f'Current time step @t: {t}')
            print(f'Changed Link Prediction task by AUC score: use current emb @t to predict **future** changed links @t+1')
            pos_edges_with_label, neg_edges_with_label = gen_test_edge_wrt_changes(G_dynamic[t],G_dynamic[t+1]) # use current emb @t predict graph t+1
            test_edges = [e[:2] for e in pos_edges_with_label] + [e[:2] for e in neg_edges_with_label]
            test_label = [e[2] for e in pos_edges_with_label] + [e[2] for e in neg_edges_with_label]
            ds_task = lpClassifier(emb_dict=emb_dicts[t])  # use current emb @t
            ds_task.evaluate_auc(test_edges, test_label)

    if args.task == 'lp' or args.task == 'all':
        from libne.downstream import lpClassifier, gen_test_edge_wrt_changes_plus_others
        for t in range(len(emb_dicts)-1):
            print(f'Current time step @t: {t}')
            print(f'Link Prediction task by AUC score: use current emb @t to predict **future** changed links @t+1')
            pos_edges_with_label, neg_edges_with_label = gen_test_edge_wrt_changes_plus_others(G_dynamic[t],G_dynamic[t+1]) # use current emb @t predict graph t+1
            test_edges = [e[:2] for e in pos_edges_with_label] + [e[:2] for e in neg_edges_with_label]
            test_label = [e[2] for e in pos_edges_with_label] + [e[2] for e in neg_edges_with_label]
            ds_task = lpClassifier(emb_dict=emb_dicts[t])  # use current emb @t
            ds_task.evaluate_auc(test_edges, test_label)
            
    if args.task == 'gr_changed' or args.task == 'all':
        from libne.downstream import grClassifier, gen_test_node_wrt_changes
        for t in range(len(emb_dicts)-1):
            precision_at_k = 20
            print(f'Current time step @t: {t}')
            print(f'Changed Graph Reconstruction by MAP @{precision_at_k} task: use current emb @t to reconstruct **current** graph @t')
            test_nodes = gen_test_node_wrt_changes(G_dynamic[t],G_dynamic[t+1])
            ds_task = grClassifier(emb_dict=emb_dicts[t], rc_graph=G_dynamic[t]) # use current emb @t
            ds_task.evaluate_precision_k(top_k=precision_at_k, node_list=test_nodes) # use current emb @t reconstruct graph t
            # ds_task.evaluate_average_precision_k(top_k=precision_at_k, node_list=test_nodes)
            # NOTE: if memory error, try grClassifier_batch (see dowmstream.py) which is slow but greatly reduce ROM

    if args.task == 'gr' or args.task == 'all':
        from libne.downstream import grClassifier
        for t in range(len(emb_dicts)-1):
            precision_at_k = 20
            print(f'Current time step @t: {t}')
            print(f'Graph Reconstruction by MAP @{precision_at_k} task: use current emb @t to reconstruct **current** graph @t')
            ds_task = grClassifier(emb_dict=emb_dicts[t], rc_graph=G_dynamic[t]) # use current emb @t reconstruct graph t
            ds_task.evaluate_precision_k(top_k=precision_at_k)
            # ds_task.evaluate_average_precision_k(top_k=precision_at_k)
            # NOTE: if memory error, try grClassifier_batch (see dowmstream.py) which is slow but greatly reduce ROM
    
    if args.task == 'nc' or args.task == 'all':
        from libne.downstream import ncClassifier
        from sklearn.linear_model import LogisticRegression  # to do... try SVM...
        try:
            label_dict = load_any_obj_pkl(args.label) # ground truth label .pkl
            for t in range(len(G_dynamic)):
                print(f'Current time step @t: {t}')
                print(f'Node Classification task by F1 score: use current emb @t to infer **current* corresponding label @t')
                X = []
                Y = []
                for node in G_dynamic[t].nodes(): # only select current available nodes for eval
                    X.append(node)
                    Y.append(str(label_dict[node])) # label as str, otherwise, sklearn error
                ds_task = ncClassifier(emb_dict=emb_dicts[t], clf=LogisticRegression())  # use current emb @t
                ds_task.split_train_evaluate(X, Y, train_precent=0.5)
        except:
            print(f'ground truth label file not exist; not support node classification task')
    """
    t2 = time.time()
    print(f'STEP3: end evaluating; time cost: {(t2-t1):.2f}s')


if __name__ == '__main__':
    print(f'------ START @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')
    main(parse_args())
    print(f'------ END @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')