"""
http://konect.uni-koblenz.de/networks/dnc-temporalGraph
Size	2,029 vertices (persons)
Volume	39,264 edges (emails)
Unique volume	5,598 edges (emails)
Average degree (overall)	38.703 edges / vertex

gap = 1
the latest [-32,-1] i.e. 21 step except the last one
undirected without self-loop dynamic graphs and isolated nodes

@ graph 0 # of nodes 1019 # of edges 2274
@ graph 1 # of nodes 1079 # of edges 2419
@ graph 2 # of nodes 1110 # of edges 2568
@ graph 3 # of nodes 1162 # of edges 2704
@ graph 4 # of nodes 1169 # of edges 2731
@ graph 5 # of nodes 1176 # of edges 2752
@ graph 6 # of nodes 1219 # of edges 2910
@ graph 7 # of nodes 1273 # of edges 3044
@ graph 8 # of nodes 1308 # of edges 3145
@ graph 9 # of nodes 1349 # of edges 3266
@ graph 10 # of nodes 1381 # of edges 3355
@ graph 11 # of nodes 1392 # of edges 3387
@ graph 12 # of nodes 1399 # of edges 3399
@ graph 13 # of nodes 1441 # of edges 3534
@ graph 14 # of nodes 1496 # of edges 3669
@ graph 15 # of nodes 1584 # of edges 3854
@ graph 16 # of nodes 1625 # of edges 3952
@ graph 17 # of nodes 1653 # of edges 4062
@ graph 18 # of nodes 1684 # of edges 4119
@ graph 19 # of nodes 1699 # of edges 4147
@ graph 20 # of nodes 1847 # of edges 4330

by Chengbin Hou
"""

import networkx as nx
import datetime 
import pickle
import pandas as pd

gap = 1

def save_nx_graph(nx_graph, path='dyn_graphs_dataset.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(nx_graph, f, protocol=pickle.HIGHEST_PROTOCOL)  # the higher protocol, the smaller file

if __name__ == '__main__':
    # --- load data ---
    df = pd.read_csv('DNC.txt', sep='\t| ', names=['from','to','weight?','time'], header=None, comment='%')
    df['time'] = df['time'].apply(lambda x: int(datetime.datetime.utcfromtimestamp(x).strftime('%Y%m%d')))
    all_days = len(pd.unique(df['time']))
    print('# of all edges: ', len(df))
    print('all unique days: ', all_days)
    print(df.head(5))

    # --- check the time oder, if not ascending, resort it ---
    tmp = df['time'][0]
    for i in range(len(df['time'])):
        if df['time'][i] > tmp:
            tmp = df['time'][i]
        elif df['time'][i] == tmp:
            pass
        else:
            print('not ascending --> we resorted it')
            print(df[i-2:i+2])
            df.sort_values(by='time', ascending=True, inplace=True)
            df.reset_index(inplace=True)
            print(df[i-2:i+2])
            break
        if i == len(df['time'])-1:
            print('ALL checked --> ascending!!!')
    
    # --- generate graph and dyn_graphs ---
    cnt_graphs = 0
    graphs = []
    g = nx.Graph()
    tmp = df['time'][0]   # time is in ascending order
    for i in range(len(df['time'])):
        if tmp == df['time'][i]:        # if is in current day
            g.add_edge(str(df['from'][i]), str(df['to'][i]))
            if i == len(df['time'])-1:  # EOF ---
                cnt_graphs += 1
                # graphs.append(g.copy())  # ignore the last day
                print('processed graphs ', cnt_graphs, '/', all_days, 'ALL done......\n')
        elif tmp < df['time'][i]:       # if goes to next day
            cnt_graphs += 1
            if (cnt_graphs//gap) >= (all_days//gap-70) and cnt_graphs%gap == 0: # the last 50 graphs 'and' the gap
                g.remove_edges_from(g.selfloop_edges())
                g.remove_nodes_from(list(nx.isolates(g)))
                graphs.append(g.copy())     # append previous g; for a part of graphs to reduce ROM
                # g = nx.Graph()            # reset graph, based on the real-world application 
            if cnt_graphs % 50 == 0:
                print('processed graphs ', cnt_graphs, '/', all_days)
            tmp = df['time'][i]
            g.add_edge(str(df['from'][i]), str(df['to'][i]))
        else:
            print('ERROR -- EXIT -- please double check if time is in ascending order!')
            exit(0)
    
    # --- take out and save part of graphs ----
    print('total graphs: ', len(graphs))
    print('we take out and save the last 31 graphs......')
    graphs = graphs[-22:-1]    # the last graph has some problem... we ignore it!
    save_nx_graph(nx_graph=graphs, path='DNC.pkl')
    for i in range(len(graphs)):
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))
    print('gap', gap)

