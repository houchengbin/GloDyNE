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

@ graph 0 # of nodes 11505 # of edges 35479
@ graph 1 # of nodes 12219 # of edges 39484
@ graph 2 # of nodes 13052 # of edges 43757
@ graph 3 # of nodes 13950 # of edges 48314
@ graph 4 # of nodes 14853 # of edges 53089
@ graph 5 # of nodes 15851 # of edges 57929
@ graph 6 # of nodes 17042 # of edges 62876
@ graph 7 # of nodes 18002 # of edges 67209
@ graph 8 # of nodes 18867 # of edges 71211
@ graph 9 # of nodes 19625 # of edges 74565
@ graph 10 # of nodes 20608 # of edges 78927
@ graph 11 # of nodes 21704 # of edges 83456
@ graph 12 # of nodes 22883 # of edges 88826
@ graph 13 # of nodes 24128 # of edges 94150
@ graph 14 # of nodes 25466 # of edges 99475
@ graph 15 # of nodes 27082 # of edges 105654
@ graph 16 # of nodes 29450 # of edges 113907
@ graph 17 # of nodes 31829 # of edges 122960
@ graph 18 # of nodes 34660 # of edges 133926
@ graph 19 # of nodes 37966 # of edges 147089
@ graph 20 # of nodes 41826 # of edges 162727
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

