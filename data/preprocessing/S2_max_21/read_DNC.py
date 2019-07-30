"""
http://konect.uni-koblenz.de/networks/dnc-temporalGraph
Size	2,029 vertices (persons)
Volume	39,264 edges (emails)
Unique volume	5,598 edges (emails)
Average degree (overall)	38.703 edges / vertex

gap = 10
the latest [-32,-1] i.e. 21 step except the last one
undirected without self-loop dynamic graphs and isolated nodes

@ graph 0 # of nodes 64 # of edges 62
@ graph 1 # of nodes 73 # of edges 74
@ graph 2 # of nodes 89 # of edges 91
@ graph 3 # of nodes 103 # of edges 105
@ graph 4 # of nodes 130 # of edges 137
@ graph 5 # of nodes 164 # of edges 172
@ graph 6 # of nodes 181 # of edges 193
@ graph 7 # of nodes 194 # of edges 206
@ graph 8 # of nodes 213 # of edges 228
@ graph 9 # of nodes 223 # of edges 238
@ graph 10 # of nodes 233 # of edges 248
@ graph 11 # of nodes 256 # of edges 272
@ graph 12 # of nodes 266 # of edges 283
@ graph 13 # of nodes 289 # of edges 307
@ graph 14 # of nodes 305 # of edges 323
@ graph 15 # of nodes 321 # of edges 339
@ graph 16 # of nodes 328 # of edges 347
@ graph 17 # of nodes 340 # of edges 359
@ graph 18 # of nodes 343 # of edges 363
@ graph 19 # of nodes 836 # of edges 1718
@ graph 20 # of nodes 1176 # of edges 2752
gap 10

by Chengbin Hou
"""

import networkx as nx
import datetime 
import pickle
import pandas as pd

gap = 10

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

