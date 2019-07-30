"""
http://konect.uni-koblenz.de/networks/dnc-temporalGraph
Size	2,029 vertices (persons)
Volume	39,264 edges (emails)
Unique volume	5,598 edges (emails)
Average degree (overall)	38.703 edges / vertex

gap = 1
the latest [-32,-1] i.e. 21 step except the last one
undirected without self-loop dynamic graphs and isolated nodes

@ graph 0 # of nodes 259 # of edges 276
@ graph 1 # of nodes 260 # of edges 277
@ graph 2 # of nodes 261 # of edges 278
@ graph 3 # of nodes 265 # of edges 282
@ graph 4 # of nodes 266 # of edges 283
@ graph 5 # of nodes 270 # of edges 287
@ graph 6 # of nodes 270 # of edges 287
@ graph 7 # of nodes 277 # of edges 295
@ graph 8 # of nodes 285 # of edges 303
@ graph 9 # of nodes 285 # of edges 303
@ graph 10 # of nodes 285 # of edges 303
@ graph 11 # of nodes 285 # of edges 303
@ graph 12 # of nodes 286 # of edges 304
@ graph 13 # of nodes 289 # of edges 307
@ graph 14 # of nodes 289 # of edges 307
@ graph 15 # of nodes 291 # of edges 309
@ graph 16 # of nodes 292 # of edges 310
@ graph 17 # of nodes 293 # of edges 311
@ graph 18 # of nodes 295 # of edges 313
@ graph 19 # of nodes 297 # of edges 315
@ graph 20 # of nodes 300 # of edges 318
@ graph 21 # of nodes 303 # of edges 321
@ graph 22 # of nodes 303 # of edges 322
@ graph 23 # of nodes 305 # of edges 323
@ graph 24 # of nodes 305 # of edges 323
@ graph 25 # of nodes 311 # of edges 329
@ graph 26 # of nodes 314 # of edges 332
@ graph 27 # of nodes 319 # of edges 337
@ graph 28 # of nodes 319 # of edges 338
@ graph 29 # of nodes 319 # of edges 338
@ graph 30 # of nodes 321 # of edges 339
@ graph 31 # of nodes 321 # of edges 339
@ graph 32 # of nodes 321 # of edges 339
@ graph 33 # of nodes 321 # of edges 339
@ graph 34 # of nodes 321 # of edges 339
@ graph 35 # of nodes 321 # of edges 339
@ graph 36 # of nodes 321 # of edges 339
@ graph 37 # of nodes 322 # of edges 340
@ graph 38 # of nodes 324 # of edges 342
@ graph 39 # of nodes 324 # of edges 342
@ graph 40 # of nodes 325 # of edges 343
@ graph 41 # of nodes 325 # of edges 343
@ graph 42 # of nodes 326 # of edges 344
@ graph 43 # of nodes 328 # of edges 347
@ graph 44 # of nodes 328 # of edges 347
@ graph 45 # of nodes 328 # of edges 347
@ graph 46 # of nodes 330 # of edges 349
@ graph 47 # of nodes 330 # of edges 349
@ graph 48 # of nodes 332 # of edges 351
@ graph 49 # of nodes 332 # of edges 351
@ graph 50 # of nodes 333 # of edges 352
@ graph 51 # of nodes 333 # of edges 352
@ graph 52 # of nodes 338 # of edges 357
@ graph 53 # of nodes 338 # of edges 357
@ graph 54 # of nodes 340 # of edges 359
@ graph 55 # of nodes 340 # of edges 359
@ graph 56 # of nodes 340 # of edges 359
@ graph 57 # of nodes 340 # of edges 359
@ graph 58 # of nodes 340 # of edges 359
@ graph 59 # of nodes 342 # of edges 361
@ graph 60 # of nodes 342 # of edges 362
@ graph 61 # of nodes 342 # of edges 362
@ graph 62 # of nodes 342 # of edges 362
@ graph 63 # of nodes 343 # of edges 363
@ graph 64 # of nodes 343 # of edges 363
@ graph 65 # of nodes 344 # of edges 364
@ graph 66 # of nodes 344 # of edges 364
@ graph 67 # of nodes 345 # of edges 366
@ graph 68 # of nodes 345 # of edges 367
@ graph 69 # of nodes 373 # of edges 423
@ graph 70 # of nodes 411 # of edges 494
@ graph 71 # of nodes 595 # of edges 1066
@ graph 72 # of nodes 673 # of edges 1305
@ graph 73 # of nodes 740 # of edges 1521
@ graph 74 # of nodes 836 # of edges 1718
@ graph 75 # of nodes 879 # of edges 1876
@ graph 76 # of nodes 886 # of edges 1899
@ graph 77 # of nodes 900 # of edges 1927
@ graph 78 # of nodes 963 # of edges 2090
@ graph 79 # of nodes 1019 # of edges 2274
@ graph 80 # of nodes 1079 # of edges 2419
@ graph 81 # of nodes 1110 # of edges 2568
@ graph 82 # of nodes 1162 # of edges 2704
@ graph 83 # of nodes 1169 # of edges 2731
@ graph 84 # of nodes 1176 # of edges 2752
@ graph 85 # of nodes 1219 # of edges 2910
@ graph 86 # of nodes 1273 # of edges 3044
@ graph 87 # of nodes 1308 # of edges 3145
@ graph 88 # of nodes 1349 # of edges 3266
@ graph 89 # of nodes 1381 # of edges 3355
@ graph 90 # of nodes 1392 # of edges 3387
@ graph 91 # of nodes 1399 # of edges 3399
@ graph 92 # of nodes 1441 # of edges 3534
@ graph 93 # of nodes 1496 # of edges 3669
@ graph 94 # of nodes 1584 # of edges 3854
@ graph 95 # of nodes 1625 # of edges 3952
@ graph 96 # of nodes 1653 # of edges 4062
@ graph 97 # of nodes 1684 # of edges 4119
@ graph 98 # of nodes 1699 # of edges 4147
@ graph 99 # of nodes 1847 # of edges 4330

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
            if (cnt_graphs//gap) >= (all_days//gap-150) and cnt_graphs%gap == 0: # the last 50 graphs 'and' the gap
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
    graphs = graphs[-101:-1]    # the last graph has some problem... we ignore it!
    save_nx_graph(nx_graph=graphs, path='DNC.pkl')
    for i in range(len(graphs)):
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))
    print('gap', gap)

