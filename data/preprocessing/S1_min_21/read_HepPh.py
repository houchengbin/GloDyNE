"""
http://konect.uni-koblenz.de/networks/ca-cit-HepPh
Size	28,093 vertices (authors)
Volume	4,596,803 edges (collaborations)
Unique volume	3,148,447 edges (collaborations)
Average degree	327.26 edges / vertex

gap = 1
the latest [-32,-1] i.e. 21 step except the last one
undirected without self-loop dynamic graphs and isolated nodes

@ graph 0 # of nodes 16777 # of edges 1170710
@ graph 1 # of nodes 16780 # of edges 1170771
@ graph 2 # of nodes 16782 # of edges 1170822
@ graph 3 # of nodes 16791 # of edges 1171587
@ graph 4 # of nodes 16801 # of edges 1172555
@ graph 5 # of nodes 16838 # of edges 1183870
@ graph 6 # of nodes 16849 # of edges 1184407
@ graph 7 # of nodes 16861 # of edges 1185856
@ graph 8 # of nodes 16862 # of edges 1185880
@ graph 9 # of nodes 16862 # of edges 1185881
@ graph 10 # of nodes 16878 # of edges 1187126
@ graph 11 # of nodes 16890 # of edges 1188103
@ graph 12 # of nodes 16912 # of edges 1190107
@ graph 13 # of nodes 16926 # of edges 1191368
@ graph 14 # of nodes 16934 # of edges 1191914
@ graph 15 # of nodes 16934 # of edges 1191915
@ graph 16 # of nodes 16935 # of edges 1191943
@ graph 17 # of nodes 16941 # of edges 1192473
@ graph 18 # of nodes 16952 # of edges 1193478
@ graph 19 # of nodes 16952 # of edges 1193658
@ graph 20 # of nodes 16956 # of edges 1194307

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
    df = pd.read_csv('HepPh.txt', sep=' ', names=['from','to','weight?','time'], header=None)
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
            if (cnt_graphs//gap) >= (all_days//gap-30) and cnt_graphs%gap == 0: # the last 50 graphs 'and' the gap
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
    print('we take out and save the last 21 graphs......')
    graphs = graphs[-22:-1]    # the last graph has some problem... we ignore it!
    save_nx_graph(nx_graph=graphs, path='HepPh.pkl')
    for i in range(len(graphs)):
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))

