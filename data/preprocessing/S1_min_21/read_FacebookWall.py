"""
http://konect.uni-koblenz.de/networks/FacebookWall
Size	46,952 vertices (users)
Volume	876,993 edges (wall posts)
Unique volume	274,086 edges (wall posts)
Average degree (overall)	37.357 edges / vertex
Fill	0.00012433 edges / vertex2
Maximum degree	2,696 edges

gap = 1
the latest [-32,-1] i.e. 21 step except the last one
undirected without self-loop dynamic graphs and isolated nodes

@ graph 0 # of nodes 43745 # of edges 170631
@ graph 1 # of nodes 43869 # of edges 171123
@ graph 2 # of nodes 44035 # of edges 171714
@ graph 3 # of nodes 44157 # of edges 172272
@ graph 4 # of nodes 44257 # of edges 172824
@ graph 5 # of nodes 44395 # of edges 173514
@ graph 6 # of nodes 44517 # of edges 174277
@ graph 7 # of nodes 44620 # of edges 175003
@ graph 8 # of nodes 44725 # of edges 175700
@ graph 9 # of nodes 44818 # of edges 176370
@ graph 10 # of nodes 44898 # of edges 176908
@ graph 11 # of nodes 44970 # of edges 177447
@ graph 12 # of nodes 45068 # of edges 178087
@ graph 13 # of nodes 45168 # of edges 178781
@ graph 14 # of nodes 45277 # of edges 179462
@ graph 15 # of nodes 45366 # of edges 180116
@ graph 16 # of nodes 45459 # of edges 180767
@ graph 17 # of nodes 45539 # of edges 181302
@ graph 18 # of nodes 45591 # of edges 181770
@ graph 19 # of nodes 45668 # of edges 182370
@ graph 20 # of nodes 45751 # of edges 183004

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
    df = pd.read_csv('FacebookWall.txt', sep=' | \t', names=['from','to','weight?','time'], header=None, comment='%')
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
            if (cnt_graphs//gap) >= (all_days//gap-35) and cnt_graphs%gap == 0: # the last 50 graphs but ignore the latest 10 'and' the gap
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
    save_nx_graph(nx_graph=graphs, path='FacebookWall.pkl')
    for i in range(len(graphs)):
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))
    print('gap', gap)

