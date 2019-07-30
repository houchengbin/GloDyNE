"""
http://konect.uni-koblenz.de/networks/chess
Size	7,301 vertices (players)
Volume	65,053 edges (games)
Unique volume	34,564 edges (games)
Average degree (overall)	17.820 edges / vertex
Fill	0.00074105 edges / vertex2
Maximum degree	280 edges

gap = 1  --> 1 means 1 month
the latest [-32,-1] i.e. 21 step except the last one
undirected without self-loop dynamic graphs

@ graph 0 # of nodes 1924 # of edges 6404
@ graph 1 # of nodes 2150 # of edges 7810
@ graph 2 # of nodes 2450 # of edges 9937
@ graph 3 # of nodes 2708 # of edges 11922
@ graph 4 # of nodes 2839 # of edges 13223
@ graph 5 # of nodes 2910 # of edges 13646
@ graph 6 # of nodes 3053 # of edges 15450
@ graph 7 # of nodes 3191 # of edges 16956
@ graph 8 # of nodes 3334 # of edges 18275
@ graph 9 # of nodes 3515 # of edges 20362
@ graph 10 # of nodes 3622 # of edges 21932
@ graph 11 # of nodes 3753 # of edges 23753
@ graph 12 # of nodes 3932 # of edges 26127
@ graph 13 # of nodes 4043 # of edges 27394
@ graph 14 # of nodes 4171 # of edges 29331
@ graph 15 # of nodes 4269 # of edges 31299
@ graph 16 # of nodes 4335 # of edges 32675
@ graph 17 # of nodes 4438 # of edges 33959
@ graph 18 # of nodes 4645 # of edges 36544
@ graph 19 # of nodes 4786 # of edges 38051
@ graph 20 # of nodes 5153 # of edges 40624
gap 4

by Chengbin Hou
"""

import networkx as nx
import datetime 
import pickle
import pandas as pd

gap = 4

def save_nx_graph(nx_graph, path='dyn_graphs_dataset.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(nx_graph, f, protocol=pickle.HIGHEST_PROTOCOL)  # the higher protocol, the smaller file

if __name__ == '__main__':
    # --- load data ---
    df = pd.read_csv('Chess.txt', sep='\t| ', names=['from','to','weight?','time'], header=None, comment='%')
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
    graphs = graphs[-22:-1]
    save_nx_graph(nx_graph=graphs, path='Chess.pkl')
    for i in range(len(graphs)):
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))
    print('gap', gap)

