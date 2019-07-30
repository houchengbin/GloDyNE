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

@ graph 0 # of nodes 293 # of edges 564
@ graph 1 # of nodes 366 # of edges 811
@ graph 2 # of nodes 636 # of edges 1576
@ graph 3 # of nodes 806 # of edges 2107
@ graph 4 # of nodes 878 # of edges 2406
@ graph 5 # of nodes 1009 # of edges 2959
@ graph 6 # of nodes 1161 # of edges 3408
@ graph 7 # of nodes 1237 # of edges 3815
@ graph 8 # of nodes 1807 # of edges 5675
@ graph 9 # of nodes 1817 # of edges 5721
@ graph 10 # of nodes 1870 # of edges 6045
@ graph 11 # of nodes 1924 # of edges 6404
@ graph 12 # of nodes 1958 # of edges 6710
@ graph 13 # of nodes 2002 # of edges 6974
@ graph 14 # of nodes 2049 # of edges 7343
@ graph 15 # of nodes 2150 # of edges 7810
@ graph 16 # of nodes 2213 # of edges 8465
@ graph 17 # of nodes 2282 # of edges 8774
@ graph 18 # of nodes 2366 # of edges 9246
@ graph 19 # of nodes 2450 # of edges 9937
@ graph 20 # of nodes 2641 # of edges 11079
@ graph 21 # of nodes 2648 # of edges 11355
@ graph 22 # of nodes 2663 # of edges 11536
@ graph 23 # of nodes 2708 # of edges 11922
@ graph 24 # of nodes 2734 # of edges 12399
@ graph 25 # of nodes 2791 # of edges 12720
@ graph 26 # of nodes 2815 # of edges 12962
@ graph 27 # of nodes 2839 # of edges 13223
@ graph 28 # of nodes 2853 # of edges 13344
@ graph 29 # of nodes 2882 # of edges 13508
@ graph 30 # of nodes 2898 # of edges 13600
@ graph 31 # of nodes 2910 # of edges 13646
@ graph 32 # of nodes 2981 # of edges 14432
@ graph 33 # of nodes 3010 # of edges 14825
@ graph 34 # of nodes 3031 # of edges 15022
@ graph 35 # of nodes 3053 # of edges 15450
@ graph 36 # of nodes 3066 # of edges 15649
@ graph 37 # of nodes 3096 # of edges 15864
@ graph 38 # of nodes 3144 # of edges 16268
@ graph 39 # of nodes 3191 # of edges 16956
@ graph 40 # of nodes 3221 # of edges 17250
@ graph 41 # of nodes 3253 # of edges 17573
@ graph 42 # of nodes 3292 # of edges 17931
@ graph 43 # of nodes 3334 # of edges 18275
@ graph 44 # of nodes 3356 # of edges 18737
@ graph 45 # of nodes 3447 # of edges 19569
@ graph 46 # of nodes 3477 # of edges 19852
@ graph 47 # of nodes 3515 # of edges 20362
@ graph 48 # of nodes 3543 # of edges 20744
@ graph 49 # of nodes 3568 # of edges 21071
@ graph 50 # of nodes 3601 # of edges 21428
@ graph 51 # of nodes 3622 # of edges 21932
@ graph 52 # of nodes 3674 # of edges 22431
@ graph 53 # of nodes 3705 # of edges 22890
@ graph 54 # of nodes 3717 # of edges 23158
@ graph 55 # of nodes 3753 # of edges 23753
@ graph 56 # of nodes 3835 # of edges 24581
@ graph 57 # of nodes 3915 # of edges 25668
@ graph 58 # of nodes 3923 # of edges 25919
@ graph 59 # of nodes 3932 # of edges 26127
@ graph 60 # of nodes 3940 # of edges 26317
@ graph 61 # of nodes 3970 # of edges 26621
@ graph 62 # of nodes 4015 # of edges 27122
@ graph 63 # of nodes 4043 # of edges 27394
@ graph 64 # of nodes 4083 # of edges 27906
@ graph 65 # of nodes 4110 # of edges 28417
@ graph 66 # of nodes 4150 # of edges 28901
@ graph 67 # of nodes 4171 # of edges 29331
@ graph 68 # of nodes 4206 # of edges 30121
@ graph 69 # of nodes 4242 # of edges 30922
@ graph 70 # of nodes 4243 # of edges 31060
@ graph 71 # of nodes 4269 # of edges 31299
@ graph 72 # of nodes 4281 # of edges 31501
@ graph 73 # of nodes 4288 # of edges 31578
@ graph 74 # of nodes 4294 # of edges 31975
@ graph 75 # of nodes 4335 # of edges 32675
@ graph 76 # of nodes 4359 # of edges 32891
@ graph 77 # of nodes 4389 # of edges 33107
@ graph 78 # of nodes 4408 # of edges 33315
@ graph 79 # of nodes 4438 # of edges 33959
@ graph 80 # of nodes 4473 # of edges 34460
@ graph 81 # of nodes 4529 # of edges 35480
@ graph 82 # of nodes 4551 # of edges 35856
@ graph 83 # of nodes 4645 # of edges 36544
@ graph 84 # of nodes 4658 # of edges 36754
@ graph 85 # of nodes 4726 # of edges 37029
@ graph 86 # of nodes 4760 # of edges 37701
@ graph 87 # of nodes 4786 # of edges 38051
@ graph 88 # of nodes 4826 # of edges 38572
@ graph 89 # of nodes 4861 # of edges 38973
@ graph 90 # of nodes 4897 # of edges 39414
@ graph 91 # of nodes 5153 # of edges 40624
@ graph 92 # of nodes 5645 # of edges 42916
@ graph 93 # of nodes 5801 # of edges 44455
@ graph 94 # of nodes 6187 # of edges 46395
@ graph 95 # of nodes 6554 # of edges 48288
@ graph 96 # of nodes 6806 # of edges 50447
@ graph 97 # of nodes 7045 # of edges 52344
@ graph 98 # of nodes 7181 # of edges 54335
@ graph 99 # of nodes 7301 # of edges 55899

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
                graphs.append(g.copy())  # ignore the last day
                print('processed graphs ', cnt_graphs, '/', all_days, 'ALL done......\n')
        elif tmp < df['time'][i]:       # if goes to next day
            cnt_graphs += 1
            if (cnt_graphs//gap) >= (all_days//gap-1000) and cnt_graphs%gap == 0: # the last 50 graphs 'and' the gap
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
    graphs = graphs[:]
    save_nx_graph(nx_graph=graphs, path='Chess.pkl')
    for i in range(len(graphs)):
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))
    print('gap', gap)

