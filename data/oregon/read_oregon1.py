"""
J. Leskovec, J. Kleinberg and C. Faloutsos. Graphs over Time: Densification Laws, Shrinking Diameters and Possible Explanations. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2005.

"""

import numpy as np
import networkx as nx
import datetime
import os
import pickle
import matplotlib.pyplot as plt

"""
The dataset contains 733 daily instances which span an interval of 785 days from November 8 1997 to January 2 2000
"""




def generate_dynamic_graph():#since there is only 9 graphs. call the function and
    file_names = ['oregon1_010331.txt',
                  'oregon1_010407.txt',
                  'oregon1_010414.txt',
                  'oregon1_010421.txt',
                  'oregon1_010428.txt',
                  'oregon1_010505.txt',
                  'oregon1_010512.txt',
                  'oregon1_010519.txt',
                  'oregon1_010526.txt',]
    graphs = []
    for i in range(9):
        print('generate graphs:',i)
        graph = generate_a_graph(file_names[i])
        graphs.append(graph)
        #print('start to draw graph:',i)
        print('graph node size:',len(graph.nodes()))
        print('graph edge size',len(graph.edges()))
        #nx.draw_networkx(graph)
        #print('show graph:',i)
        #plt.show(graph)
    return graphs


def generate_a_graph(file_name):

    graph_data = np.genfromtxt(file_name, dtype=str)
    graph = nx.Graph()

    for i in range(len(graph_data)):
        graph.add_edge(graph_data[i][0],graph_data[i][1])
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
    graphs = generate_dynamic_graph()
    #save_nx_graph(nx_graph=graphs, path='oregon1_dyn_graphs.pkl')
