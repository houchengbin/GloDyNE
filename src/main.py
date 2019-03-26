'''
demo of node embedding in dynamic environment: DynWalks and its competitors
STEP1: prepare data --> (input data using all graphs at a time; but note DynWalks can easily support streaming graphs/edges if possible)
STEP2: learn node embeddings -->
STEP3: downstream evaluations

python src/main.py --method DynWalks --task all --graph data/cora/cora_dyn_graphs.pkl --label data/cora/cora_node_label_dict.pkl --emb-file output/cora_DynWalks_128_embs.pkl --num-walks 20 --limit 0.05 --scheme 3 --emb-dim 100 --workers 6

DynWalks hyper-parameters:
scheme=3, limit=0.05,                       # DynWalks key hyper-parameters
restart_prob=0.0, update_threshold=0.1,     # DynWalks key hyper-parameters
num_walks=20, walk_length=80,               # deepwalk hyper-parameters
window=10, negative=5,                      # Skig-Gram hyper-parameters
seed=2019, workers=20,                      # should not related to accuracy

by Chengbin HOU <chengbin.hou10@foxmail.com>
'''

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import networkx as nx
from libne.utils import load_any_obj_pkl, save_any_obj_pkl


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    # -----------------------------------------------general settings--------------------------------------------------
    parser.add_argument('--graph', default='data/cora/cora_dyn_graphs.pkl',
                        help='graph/network file')
    parser.add_argument('--label', default='data/cora/cora_node_label_dict.pkl',
                        help='node label file')
    parser.add_argument('--emb-dim', default=128, type=int,
                        help='node embeddings dimensions')
    parser.add_argument('--task', default='save', choices=['lp', 'gc','lp_changed', 'gc_changed', ' nc', 'gr', 'all', 'save'],
                        help='choices of downstream tasks: lp, nc, gr, all, save')
    parser.add_argument('--emb-file', default='output/cora_DynWalks_128_embs.pkl',
                        help='node embeddings file; suggest: data_method_dim_embs.pkl')
    # -------------------------------------------------method settings-----------------------------------------------------------
    parser.add_argument('--method', default='DynWalks', choices=['DynWalks', 'DynWalks_noacc', 'DeepWalk', 'GraRep', 'HOPE'],
                        help='choices of Network Embedding methods')
    parser.add_argument('--restart-prob', default=None, type=float,
                        help='restart probability for random walks; raning [0.0, 1.0] or None')
    parser.add_argument('--update-threshold', default=0.1, type=float,
                        help='if changes/degree > update_threshold; update node; raning [0.0, 1.0]')
    parser.add_argument('--limit', default=0.05, type=float,
                        help='the limit of nodes to be updated at each time step')
    parser.add_argument('--scheme', default=3, type=int,
                        help='scheme 1: new+random; scheme 2: new + affected + random; scheme 3: new + affected + diverse_random')
    # walk based methods
    parser.add_argument('--num-walks', default=20, type=int,
                        help='# of random walks of each node')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='length of each random walk')
    # gensim word2vec parameters
    parser.add_argument('--window', default=10, type=int,
                        help='window size of SGNS model')
    parser.add_argument('--negative', default=5, type=int,
                        help='negative samples of SGNS model')
    parser.add_argument('--workers', default=24, type=int,
                        help='# of parallel processes.')
    parser.add_argument('--seed', default=2019, type=int,
                        help='random seed')
    # walk based methods
    parser.add_argument('--Kstep', default=4, type=int,
                        help='Kstep used in GraRep model, error if not emb_dim % Kstep == 0')
    args = parser.parse_args()
    return args


def main(args):
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print(f'Summary of all settings: {args}')

    # ---------------------------------------STEP1: prepare data-----------------------------------------------------
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
        model = DynWalks.DynWalks(G_dynamic=G_dynamic, restart_prob=args.restart_prob, update_threshold=args.update_threshold,
                                    emb_dim=args.emb_dim, num_walks=args.num_walks, walk_length=args.walk_length, window=args.window, 
                                    workers=args.workers, negative=args.negative, seed=args.seed, limit=args.limit, scheme=args.scheme)
        model.sampling_traning()
    elif args.method == 'DynWalks_noacc':
        from libne import DynWalks_noacc  
        model = DynWalks_noacc.DynWalks_noacc(G_dynamic=G_dynamic, restart_prob=args.restart_prob, update_threshold=args.update_threshold,
                                    emb_dim=args.emb_dim, num_walks=args.num_walks, walk_length=args.walk_length, window=args.window, 
                                    workers=args.workers, negative=args.negative, seed=args.seed, limit=args.limit, scheme=args.scheme)
        model.sampling_traning()
    elif args.method == 'DeepWalk':
        from libne import DeepWalk 
        model = DeepWalk.DeepWalk(G_dynamic=G_dynamic, emb_dim=args.emb_dim, num_walks=args.num_walks, walk_length=args.walk_length, 
                                    window=args.window, workers=args.workers, negative=args.negative, restart_prob=None, seed=args.seed)
        model.sampling_traning()
    elif args.method == 'GraRep':
        from libne import GraRep
        model = GraRep.GraRep(G_dynamic=G_dynamic, emb_dim=args.emb_dim, Kstep=args.Kstep)
        model.traning()
    elif args.method == 'HOPE':
        from libne import HOPE
        model = HOPE.HOPE(G_dynamic=G_dynamic, emb_dim=args.emb_dim)
        model.traning()
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

    t2 = time.time()
    print(f'STEP3: end evaluating; time cost: {(t2-t1):.2f}s')


if __name__ == '__main__':
    print(f'------ START @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')
    main(parse_args())
    print(f'------ END @ {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())} ------')
