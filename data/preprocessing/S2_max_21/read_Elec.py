"""
http://konect.uni-koblenz.de/networks/elec
Size	7,118 vertices (users)
Volume	103,675 edges (votes)
Average degree (overall)	29.130 edges / vertex
Fill	0.0020462 edges / vertex2
Maximum degree	1,167 edges

gap = 1
the latest [-32,-1] i.e. 21 step except the last one
undirected without self-loop dynamic graphs and isolated nodes

@ graph 0 # of nodes 2888 # of edges 31818
@ graph 1 # of nodes 3131 # of edges 35547
@ graph 2 # of nodes 3374 # of edges 39689
@ graph 3 # of nodes 3569 # of edges 43474
@ graph 4 # of nodes 3775 # of edges 46987
@ graph 5 # of nodes 3993 # of edges 50189
@ graph 6 # of nodes 4163 # of edges 52642
@ graph 7 # of nodes 4389 # of edges 55820
@ graph 8 # of nodes 4539 # of edges 58496
@ graph 9 # of nodes 4707 # of edges 60799
@ graph 10 # of nodes 4886 # of edges 63789
@ graph 11 # of nodes 5101 # of edges 67737
@ graph 12 # of nodes 5311 # of edges 70958
@ graph 13 # of nodes 5542 # of edges 74017
@ graph 14 # of nodes 5771 # of edges 78039
@ graph 15 # of nodes 5985 # of edges 81520
@ graph 16 # of nodes 6148 # of edges 84164
@ graph 17 # of nodes 6325 # of edges 86905
@ graph 18 # of nodes 6487 # of edges 89481
@ graph 19 # of nodes 6614 # of edges 92028
@ graph 20 # of nodes 6833 # of edges 96213
gap 30

by Chengbin Hou
"""

import networkx as nx
import datetime 
import pickle
import pandas as pd

gap = 30

def save_nx_graph(nx_graph, path='dyn_graphs_dataset.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(nx_graph, f, protocol=pickle.HIGHEST_PROTOCOL)  # the higher protocol, the smaller file

if __name__ == '__main__':
    # --- load data ---
    df = pd.read_csv('Elec.txt', sep='\t| ', names=['from','to','weight?','time'], header=None, comment='%')
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
    print('we take out and save the last 21 graphs......')
    graphs = graphs[-22:-1]    # the last graph has some problem... we ignore it!
    save_nx_graph(nx_graph=graphs, path='Elec.pkl')
    for i in range(len(graphs)):
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))
    print('gap', gap)

