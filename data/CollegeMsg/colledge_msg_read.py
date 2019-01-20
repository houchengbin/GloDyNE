"""
data set source
Pietro Panzarasa, Tore Opsahl, and Kathleen M. Carley. "Patterns and dynamics of users' behavior and interaction: Network analysis of an online community." Journal of the American Society for Information Science and Technology 60.5 (2009): 911-932.
"""

import numpy as np
import networkx as nx
from bisect import bisect
import matplotlib.pyplot as plt
import pickle

def read_txt_file():
    #read the txt file
    lines = np.genfromtxt("CollegeMsg.txt", dtype= int)
    return lines


def get_colledge_msg_data(no_new_node_appear = False, edge_last_n_time_step = 0, time_step_number = 5):
    """
    :param no_new_node_appear: True
    :param edge_last_n_time_step:
    :param time_step_number:
    :return:
    """
    # a interval will be provided if True
    data = read_txt_file()#edges
    sorted_data = data[data[:, 2].argsort()]
    """
    for i in range(len(sorted_data)):
        print(sorted_data[i])
    """
    graphs = get_time_based_networks(sorted_data, time_step_number, edge_last_n_time_step, no_new_node_appear)
    return graphs

def get_time_based_networks(sorted_data,time_step_number, edge_last_n_time_step, no_new_node_appear):

    all_nodes = set(sorted_data[:, 0]).union(set(sorted_data[:, 1]))
    all_nodes = list(all_nodes)

    #print(len(all_nodes), " ",all_nodes)#shou be 1899
    #all_nodes_no_duplicate = set(all_nodes)

    earliest_time = sorted_data[0][2]
    latest_time = sorted_data[len(sorted_data)-1][2]
    time_gap = float(latest_time - earliest_time)/(time_step_number+1)#i time slot require i+1 time point
    graphs = []
    blank_graph = nx.Graph()
    time_step_slots = []
    #time_step_slots.append(earliest_time)

    for i in range(1,time_step_number):
        time_step_slots.append(earliest_time + i*time_gap)

    time_step_slots.append(latest_time+1)

    if no_new_node_appear == True:#change
        for i in range(len(all_nodes)):
            blank_graph.add_node(str(all_nodes[i]))

    for i in range(time_step_number):
        graphs.append(blank_graph.copy())


    for i in range(len(sorted_data)):
        edge_time_tag = sorted_data[i][2]
        calculated_time_step = bisect(time_step_slots, edge_time_tag)
        #print(time_step_slots)
        # print(edge_time_tag)
        # print("processing edge:",i, " total edge:",len(sorted_data), "calculated_time_step:",calculated_time_step )

        if edge_last_n_time_step == 0:
            for j in range(calculated_time_step, time_step_number):
                graphs[j].add_edge((str(sorted_data[i][0])), str(sorted_data[i][1]))
        else:
            edge_disappear = calculated_time_step + edge_last_n_time_step
            if edge_disappear > time_step_number:
                edge_disappear = time_step_number
            for j in range(calculated_time_step, edge_disappear):
                graphs[j].add_edge((str(sorted_data[i][0])), str(sorted_data[i][1]))

    return graphs


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


def save_any_obj(obj, path='obj_temp.data'):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    graphs = get_colledge_msg_data()
    save_nx_graph(nx_graph=graphs, path='colledge_msg_dynamic_graphs.data')
    for i in range(len(graphs)):
        print(len(graphs[i].edges()), "edges graph:",i)
        print(len(graphs[i].nodes()), "nodes graph:",i)
        """
        print("drawing,",i)
        nx.draw_networkx(graphs[i])
        print("show ",i)
        plt.show()
        """
    
