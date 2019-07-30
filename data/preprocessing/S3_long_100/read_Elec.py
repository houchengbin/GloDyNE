"""
http://konect.uni-koblenz.de/networks/elec
Size	7,118 vertices (users)
Volume	103,675 edges (votes)
Average degree (overall)	29.130 edges / vertex
Fill	0.0020462 edges / vertex2
Maximum degree	1,167 edges

gap = 1
the latest [-32,-1] i.e. 181 step except the last one
undirected without self-loop dynamic graphs and isolated nodes

@ graph 0 # of nodes 6538 # of edges 90527
@ graph 1 # of nodes 6541 # of edges 90592
@ graph 2 # of nodes 6542 # of edges 90634
@ graph 3 # of nodes 6551 # of edges 90784
@ graph 4 # of nodes 6556 # of edges 90878
@ graph 5 # of nodes 6558 # of edges 90930
@ graph 6 # of nodes 6563 # of edges 91008
@ graph 7 # of nodes 6566 # of edges 91126
@ graph 8 # of nodes 6575 # of edges 91292
@ graph 9 # of nodes 6577 # of edges 91345
@ graph 10 # of nodes 6579 # of edges 91426
@ graph 11 # of nodes 6585 # of edges 91521
@ graph 12 # of nodes 6588 # of edges 91621
@ graph 13 # of nodes 6596 # of edges 91715
@ graph 14 # of nodes 6598 # of edges 91766
@ graph 15 # of nodes 6600 # of edges 91815
@ graph 16 # of nodes 6607 # of edges 91878
@ graph 17 # of nodes 6609 # of edges 91939
@ graph 18 # of nodes 6613 # of edges 92003
@ graph 19 # of nodes 6614 # of edges 92028
@ graph 20 # of nodes 6617 # of edges 92044
@ graph 21 # of nodes 6623 # of edges 92141
@ graph 22 # of nodes 6623 # of edges 92204
@ graph 23 # of nodes 6625 # of edges 92259
@ graph 24 # of nodes 6631 # of edges 92327
@ graph 25 # of nodes 6638 # of edges 92396
@ graph 26 # of nodes 6645 # of edges 92440
@ graph 27 # of nodes 6652 # of edges 92515
@ graph 28 # of nodes 6659 # of edges 92607
@ graph 29 # of nodes 6664 # of edges 92727
@ graph 30 # of nodes 6673 # of edges 92903
@ graph 31 # of nodes 6682 # of edges 93025
@ graph 32 # of nodes 6687 # of edges 93168
@ graph 33 # of nodes 6698 # of edges 93343
@ graph 34 # of nodes 6708 # of edges 93543
@ graph 35 # of nodes 6715 # of edges 93774
@ graph 36 # of nodes 6721 # of edges 93918
@ graph 37 # of nodes 6727 # of edges 94072
@ graph 38 # of nodes 6735 # of edges 94260
@ graph 39 # of nodes 6745 # of edges 94417
@ graph 40 # of nodes 6750 # of edges 94562
@ graph 41 # of nodes 6762 # of edges 94762
@ graph 42 # of nodes 6772 # of edges 95055
@ graph 43 # of nodes 6780 # of edges 95224
@ graph 44 # of nodes 6797 # of edges 95420
@ graph 45 # of nodes 6803 # of edges 95575
@ graph 46 # of nodes 6813 # of edges 95782
@ graph 47 # of nodes 6821 # of edges 95931
@ graph 48 # of nodes 6827 # of edges 96066
@ graph 49 # of nodes 6833 # of edges 96213
@ graph 50 # of nodes 6836 # of edges 96307
@ graph 51 # of nodes 6838 # of edges 96369
@ graph 52 # of nodes 6844 # of edges 96432
@ graph 53 # of nodes 6848 # of edges 96508
@ graph 54 # of nodes 6851 # of edges 96602
@ graph 55 # of nodes 6860 # of edges 96686
@ graph 56 # of nodes 6865 # of edges 96749
@ graph 57 # of nodes 6868 # of edges 96817
@ graph 58 # of nodes 6875 # of edges 96903
@ graph 59 # of nodes 6878 # of edges 96966
@ graph 60 # of nodes 6882 # of edges 97011
@ graph 61 # of nodes 6887 # of edges 97061
@ graph 62 # of nodes 6888 # of edges 97096
@ graph 63 # of nodes 6895 # of edges 97134
@ graph 64 # of nodes 6896 # of edges 97168
@ graph 65 # of nodes 6899 # of edges 97214
@ graph 66 # of nodes 6904 # of edges 97299
@ graph 67 # of nodes 6912 # of edges 97394
@ graph 68 # of nodes 6923 # of edges 97567
@ graph 69 # of nodes 6927 # of edges 97691
@ graph 70 # of nodes 6934 # of edges 97812
@ graph 71 # of nodes 6949 # of edges 97972
@ graph 72 # of nodes 6956 # of edges 98075
@ graph 73 # of nodes 6973 # of edges 98229
@ graph 74 # of nodes 6978 # of edges 98375
@ graph 75 # of nodes 6987 # of edges 98558
@ graph 76 # of nodes 6996 # of edges 98714
@ graph 77 # of nodes 7004 # of edges 98829
@ graph 78 # of nodes 7012 # of edges 98894
@ graph 79 # of nodes 7015 # of edges 98972
@ graph 80 # of nodes 7019 # of edges 99031
@ graph 81 # of nodes 7026 # of edges 99164
@ graph 82 # of nodes 7035 # of edges 99238
@ graph 83 # of nodes 7039 # of edges 99306
@ graph 84 # of nodes 7044 # of edges 99361
@ graph 85 # of nodes 7046 # of edges 99444
@ graph 86 # of nodes 7051 # of edges 99527
@ graph 87 # of nodes 7055 # of edges 99563
@ graph 88 # of nodes 7060 # of edges 99619
@ graph 89 # of nodes 7062 # of edges 99662
@ graph 90 # of nodes 7065 # of edges 99705
@ graph 91 # of nodes 7071 # of edges 99785
@ graph 92 # of nodes 7075 # of edges 99912
@ graph 93 # of nodes 7077 # of edges 100015
@ graph 94 # of nodes 7080 # of edges 100063
@ graph 95 # of nodes 7084 # of edges 100134
@ graph 96 # of nodes 7087 # of edges 100225
@ graph 97 # of nodes 7094 # of edges 100319
@ graph 98 # of nodes 7100 # of edges 100417
@ graph 99 # of nodes 7107 # of edges 100547

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
            if (cnt_graphs//gap) >= (all_days//gap-190) and cnt_graphs%gap == 0: # the last 50 graphs 'and' the gap
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
    graphs = graphs[-101:-1]    # the last graph has some problem... we ignore it!
    save_nx_graph(nx_graph=graphs, path='Elec.pkl')
    for i in range(len(graphs)):
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))
    print('gap', gap)

