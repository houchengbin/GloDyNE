"""
https://snap.stanford.edu/data/as-733.html

The dataset contains 733 daily instances which span an interval of 785 days from November 8 1997 to January 2 2000

max nodes 6467; max edges 13895
gap = 1
the latest [-32,-1] i.e. 21 step except the last one; NOTE: the last date is 20000102
undirected without self-loop dynamic graphs and isolated nodes


from 19990824 --> 100 steps
@ graph 0 # of nodes 5600 # of edges 10754
@ graph 1 # of nodes 5608 # of edges 10762
@ graph 2 # of nodes 5619 # of edges 10770
@ graph 3 # of nodes 5615 # of edges 10765
@ graph 4 # of nodes 5627 # of edges 10800
@ graph 5 # of nodes 103 # of edges 239
@ graph 6 # of nodes 5633 # of edges 10802
@ graph 7 # of nodes 3263 # of edges 6308
@ graph 8 # of nodes 5654 # of edges 10852
@ graph 9 # of nodes 5654 # of edges 10823
@ graph 10 # of nodes 5666 # of edges 10861
@ graph 11 # of nodes 5663 # of edges 10835
@ graph 12 # of nodes 5665 # of edges 10840
@ graph 13 # of nodes 5665 # of edges 10852
@ graph 14 # of nodes 5667 # of edges 10865
@ graph 15 # of nodes 5689 # of edges 10870
@ graph 16 # of nodes 5710 # of edges 10901
@ graph 17 # of nodes 5705 # of edges 10912
@ graph 18 # of nodes 5713 # of edges 10931
@ graph 19 # of nodes 5722 # of edges 10914
@ graph 20 # of nodes 5728 # of edges 10933
@ graph 21 # of nodes 5740 # of edges 10936
@ graph 22 # of nodes 5746 # of edges 11017
@ graph 23 # of nodes 5751 # of edges 11059
@ graph 24 # of nodes 5738 # of edges 11081
@ graph 25 # of nodes 5757 # of edges 11106
@ graph 26 # of nodes 5752 # of edges 11096
@ graph 27 # of nodes 5762 # of edges 11157
@ graph 28 # of nodes 5784 # of edges 11188
@ graph 29 # of nodes 5793 # of edges 11212
@ graph 30 # of nodes 5803 # of edges 11234
@ graph 31 # of nodes 5816 # of edges 11264
@ graph 32 # of nodes 5819 # of edges 11258
@ graph 33 # of nodes 5819 # of edges 11268
@ graph 34 # of nodes 5827 # of edges 11289
@ graph 35 # of nodes 5829 # of edges 11305
@ graph 36 # of nodes 5840 # of edges 11308
@ graph 37 # of nodes 5844 # of edges 11326
@ graph 38 # of nodes 5855 # of edges 11323
@ graph 39 # of nodes 5861 # of edges 11313
@ graph 40 # of nodes 5860 # of edges 11321
@ graph 41 # of nodes 5865 # of edges 11358
@ graph 42 # of nodes 5876 # of edges 11360
@ graph 43 # of nodes 5881 # of edges 11395
@ graph 44 # of nodes 5893 # of edges 11386
@ graph 45 # of nodes 5900 # of edges 11425
@ graph 46 # of nodes 5895 # of edges 11429
@ graph 47 # of nodes 5896 # of edges 11424
@ graph 48 # of nodes 5902 # of edges 11428
@ graph 49 # of nodes 5925 # of edges 11444
@ graph 50 # of nodes 5915 # of edges 11427
@ graph 51 # of nodes 5924 # of edges 11456
@ graph 52 # of nodes 5939 # of edges 11486
@ graph 53 # of nodes 5953 # of edges 11518
@ graph 54 # of nodes 6023 # of edges 11682
@ graph 55 # of nodes 6036 # of edges 11715
@ graph 56 # of nodes 6039 # of edges 11725
@ graph 57 # of nodes 6072 # of edges 11947
@ graph 58 # of nodes 6117 # of edges 12005
@ graph 59 # of nodes 6127 # of edges 12046
@ graph 60 # of nodes 3962 # of edges 7931
@ graph 61 # of nodes 3912 # of edges 7717
@ graph 62 # of nodes 6176 # of edges 12173
@ graph 63 # of nodes 2774 # of edges 5673
@ graph 64 # of nodes 6202 # of edges 12170
@ graph 65 # of nodes 6195 # of edges 12162
@ graph 66 # of nodes 6209 # of edges 12206
@ graph 67 # of nodes 6214 # of edges 12232
@ graph 68 # of nodes 6232 # of edges 12216
@ graph 69 # of nodes 6235 # of edges 12100
@ graph 70 # of nodes 6243 # of edges 12113
@ graph 71 # of nodes 6296 # of edges 12202
@ graph 72 # of nodes 6289 # of edges 12168
@ graph 73 # of nodes 6301 # of edges 12226
@ graph 74 # of nodes 767 # of edges 1734
@ graph 75 # of nodes 1470 # of edges 3131
@ graph 76 # of nodes 1486 # of edges 3172
@ graph 77 # of nodes 1477 # of edges 3142
@ graph 78 # of nodes 1476 # of edges 3132
@ graph 79 # of nodes 2071 # of edges 4233
@ graph 80 # of nodes 2086 # of edges 4283
@ graph 81 # of nodes 2062 # of edges 4233
@ graph 82 # of nodes 2090 # of edges 4289
@ graph 83 # of nodes 2070 # of edges 4240
@ graph 84 # of nodes 2073 # of edges 4241
@ graph 85 # of nodes 2102 # of edges 4307
@ graph 86 # of nodes 2067 # of edges 4218
@ graph 87 # of nodes 2080 # of edges 4271
@ graph 88 # of nodes 2095 # of edges 4291
@ graph 89 # of nodes 2083 # of edges 4263
@ graph 90 # of nodes 2058 # of edges 4227
@ graph 91 # of nodes 2063 # of edges 4233
@ graph 92 # of nodes 2092 # of edges 4285
@ graph 93 # of nodes 2089 # of edges 4270
@ graph 94 # of nodes 2122 # of edges 4334
@ graph 95 # of nodes 2120 # of edges 4314
@ graph 96 # of nodes 2132 # of edges 4347
@ graph 97 # of nodes 2107 # of edges 4303
@ graph 98 # of nodes 3570 # of edges 7033
@ graph 99 # of nodes 6474 # of edges 12572

by Chengbin Hou
"""

import numpy as np
import networkx as nx
import datetime
import os
import pickle


def detect_exentence_file(date):
    file_location = date_2_string(date)
    return os.path.isfile(file_location)

def date_2_string(date):#change the date format to
    year = date.year
    month = date.month
    day = date.day
    string_date = str(year*10000 + month*100 + day)
    string_date = 'as'+string_date+'.txt'
    return string_date

def string_2_date(date_str):
    year = date_str[0:4]
    month = date_str[4:6]
    day = date_str[6:8]
    date = datetime.datetime(year = int(year), month = int(month), day = int(day))
    return date


def generate_dynamic_graph(start_date = '19991009', time_step_number = 10, stop_at_irregular_interval = False):
    """
    earlist date is 19971108
    last date is 20000102

    the form of input is a string of date such as '19991015'. Note that I did not implement any date check

    :param start_date:
    :param time_step_number:
    :param stop_at_irregular_interval:
    :return:
    """
    user_chosen_date = string_2_date(start_date)
    dyanmic_netowks = []
    last_available_date = datetime.datetime(2000,1,2)


    remaining_graph = time_step_number
    while(remaining_graph > 0):
        if (user_chosen_date - last_available_date).days > 0:
            print("no more file available, stop generate more file")
            break
        elif detect_exentence_file(user_chosen_date) == True:

            remaining_graph -= 1
            file_name = date_2_string(user_chosen_date)
            graph = generate_a_graph(file_name)
            dyanmic_netowks.append(graph.copy())
            print(remaining_graph)
            #print(len(graph.nodes())," graph node number")
            user_chosen_date += datetime.timedelta(days=1)
        elif stop_at_irregular_interval == False:
            print("file does not exit at ", user_chosen_date, "date skipped")
            user_chosen_date += datetime.timedelta(days=1)
        else:
            print("file does not exit at ", user_chosen_date, "stop generate more network")
            break
    print("dynamic network length:",len(dyanmic_netowks))
    return dyanmic_netowks



def generate_a_graph(file_name):

    graph_data = np.genfromtxt(file_name, dtype=str)
    graph = nx.Graph()

    for i in range(len(graph_data)):
        graph.add_edge(str(graph_data[i][0]), str(graph_data[i][1]))
    
    graph.remove_edges_from(graph.selfloop_edges())
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph

def save_nx_graph(nx_graph, path='nx_graph_temp.data'):
    with open(path, 'wb') as f:
        pickle.dump(nx_graph, f, protocol=pickle.HIGHEST_PROTOCOL)  # the higher protocol, the smaller file
    with open(path, 'rb') as f:
        nx_graph_reload = pickle.load(f)

    try:
        print('Check if it is correctly dumped and loaded: ', nx_graph_reload.edges() == nx_graph.edges(),
              ' It contains only ONE graph')
    except:
        for i in range(len(nx_graph)):
            print('Check if it is correctly dumped and loaded: ', nx_graph_reload[i].edges() == nx_graph[i].edges(),
                  ' for Graph ', i)

if __name__ == '__main__':
    """
    start_date = datetime.datetime(1997,11,8)
    last_date = datetime.datetime(2000,1,2)
    days = (last_date-start_date).days
    print(days)
    #785 internal however which 733 days are missing
    print(date_2_string(start_date))
    detect_exentence_file()
    """
    graphs = generate_dynamic_graph(start_date='19990824', time_step_number=100, stop_at_irregular_interval=False)

    graphs = graphs[:]    # the last graph has some problem... we ignore it!
    save_nx_graph(nx_graph=graphs, path='AS733.pkl')

    for i in range(len(graphs)):
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))
