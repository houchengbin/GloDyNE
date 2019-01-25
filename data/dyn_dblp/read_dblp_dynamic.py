"""
not finished as something went wrong

"""

import numpy as np
import networkx as nx
import pickle
import datetime

def read_txt_file():
    #read the txt file
    lines = np.genfromtxt("out.dblp_coauthor.txt", dtype = int)

    names = np.genfromtxt("ent.author.txt", dtype=str)
    return lines, names

def read_txt_files():#second version

    text_file = open("out.dblp_coauthor.txt", "r")
    lines = text_file.read().split('\n')
    text_file.close()
    edges = []
    """
    for i in range(len(lines)):
        text_split = lines[i].split(' ')
        print(text_split)
        print(text_split[0],text_split[1],text_split[2])
    """

    for i in range(1,len(lines)):
        items = lines[i].split(' ')
        edges.append([int(items[0]),int(items[1]),int(items[3]),int(items[4])])

    print(len(edges)," edges size")
    name_dict = {}
    names = np.genfromtxt("ent.author.txt", dtype=str)
    for i in range(len(names)):
        name_dict[names[i][0]] = names[i][1]

    return edges, name_dict




def generate_dynamic_graph(time_step_number = 5):
    lines, name_dict = read_txt_files()
    """
    min_mum_non_zero_time = 0
    max_mum_time = 0
    for i in range(len(lines)):
        time_tag = lines[i][3]
        if time_tag < min_mum_non_zero_time:
            min_mum_non_zero_time = time_tag
        if time_tag > max_mum_time:
            max_mum_time = time_tag

    print(min_mum_non_zero_time)
    print(max_mum_time)
    print(-504921539, "min time")
    print(1388534461, "max time")
    print(len(lines)," edge set size total")
    """
    min_time = float(-504921539)
    max_time = float(1388534461)
    diff = (max_time - min_time)/float(time_step_number)
    time_slot = []

    for i in range(time_step_number-1):
        time_slot.append(min_time + diff*(1+i))
    time_slot.append(max_time+1)
    print(time_slot)
    graphs = []
    for i in range(time_step_number):
        graphs.append(nx.Graph())

    for i in range(len(lines)):
        time_tag = lines[i][3]
        showed_up_time_slot = 0
        for j in range(len(time_slot)):
            if time_tag < time_slot[j]:
                showed_up_time_slot = j
                break
        for j in range(showed_up_time_slot, len(graphs)):
            graphs[j].add_edge(str(lines[i][0]),str(lines[i][1]))


    return graphs, name_dict


def save_any_obj(obj, path='obj_temp.data'):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

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

    #graphs = generate_dynamic_graph()
    #save_nx_graph(nx_graph=graphs, path='dblp_dynamic_graphs.data')
    #print(datetime.strftime(883612861))
    graphs, name_dict = generate_dynamic_graph()
    """
    save_any_obj(obj=name_dict, path='dblp_node_name_dict.pkl')  # {node ID: degree, ...}
    save_nx_graph(nx_graph=graphs, path='dblp_dyn_graphs.pkl')
    """