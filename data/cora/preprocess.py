import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

# by Chengbin Hou

# -----------------------------------------------------------------
# ------------------------- data generator ------------------------
# -----------------------------------------------------------------

def generate_dynamic_data(initial_G, time_step=5, initial_edge_porpotation=0.5):
    # reference_graph = initial_G.copy()
    initial_nodes_number = len(initial_G.nodes())
    initial_edges_number = len(initial_G.edges())
    initial_edges = list(initial_G.edges())
    # disappear_nodes_number = int(initial_edges_number*initial_edge_porpotation)
    chance_each_time_step = (1 - initial_edge_porpotation) / time_step

    time_step_slots = []
    time_step_slots.append(initial_edge_porpotation)
    for i in range(time_step):
        time_step_slots.append(initial_edge_porpotation + (i + 1) * chance_each_time_step)

    edges_time_step = []
    for i in range(initial_edges_number):
        random_number = random.random()
        for j in range(len(time_step_slots)):
            if time_step_slots[j] > random_number:
                edges_time_step.append(j)
                break

    graphs = []
    for i in range(time_step + 1):
        graphs.append(nx.Graph())
    for i in range(len(edges_time_step)):
        current_edge_time_step = edges_time_step[i]
        for j in range(current_edge_time_step, len(graphs)):
            graphs[j].add_edge(str(list(initial_edges[i])[0]), str(list(initial_edges[i])[1]))

    for i in range(len(graphs)):
        #nx.draw_networkx(graphs[i])
        if i > 0:
            print('nx.is_directed(G)?', nx.is_directed(graphs[i]))
            # print("edge deleted v1", set(graphs[i - 1].edges()) - set(graphs[i].edges()))
            # print("edge deleted v2", edge_s1_minus_s0(graphs[i-1],graphs[i]))
            # print("edge added v1", set(graphs[i].edges()) - set(graphs[i - 1].edges()))
            # print("edge added v2", edge_s1_minus_s0(graphs[i],graphs[i-1]))
            # print("node deleted", set(graphs[i].nodes()) - set(graphs[i-1].nodes()))
            # print("node added", set(graphs[i].nodes()) - set(graphs[i - 1].nodes()))

        print("graph_size: ", len(graphs[i]), '====== @ time step', i)

        #plt.show(graphs[i])

    return graphs

# ----------------------------------------------------------------
# --------------------------- utils ------------------------------
# ----------------------------------------------------------------
def edge_s1_minus_s0(s1, s0, is_directed=False):
    if not is_directed:
        s1_reordered = set((a, b) if int(a) < int(b) else (b, a) for a, b in s1.edges())
        s0_reordered = set((a, b) if int(a) < int(b) else (b, a) for a, b in s0.edges())
        return s1_reordered - s0_reordered
    else:
        print('currently not support directed case')

def unique_nodes_from_edge_set(edge_set):
    unique_nodes = []
    for a, b in edge_set:
        if a not in unique_nodes:
            unique_nodes.append(a)
        if b not in unique_nodes:
            unique_nodes.append(b)
    return unique_nodes

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



def read_node_label_downstream(path):
    """ may be used in node classification task;
        part of labels for training clf and
        the result served as ground truth;
        note: similar method can be found in graph.py -> read_node_label
    """
    node_label_dict = {}
    fin = open(path, 'r')
    while 1:
        line = fin.readline()
        if line == '':
            break
        vec = line.strip().split(' ')
        node_label_dict[vec[0]] = vec[1:]
    fin.close()
    return node_label_dict

# --------------------------------------------------
# -------------------- test  -----------------------
# --------------------------------------------------
if __name__ == '__main__':
    G = nx.read_adjlist(path='cora_adjlist.txt', create_using=nx.Graph())
    # G = nx.read_adjlist(path='cora_doub_adjlist.txt', create_using=nx.DiGraph())

    node_label_dict = read_node_label_downstream(path='cora_label.txt')
    save_any_obj(obj=node_label_dict, path='cora_node_label_dict.pkl')  # {node ID: degree, ...}

    Gs = generate_dynamic_data(G)
    print('len(Gs[-1].nodes())', len(Gs[-1].nodes()))
    adjmatrix = nx.to_numpy_array(G)
    print('Is the graph symmetric? i.e. undirected graph?', (G==np.transpose(G)).all(), ' -- note the diff of edge # between directed and undirected nx graph')
    print('len(Gs[-1].edges())', len(Gs[-1].edges()))
    print('np.sum(adjmatrix)', np.sum(adjmatrix))

    save_nx_graph(nx_graph=Gs, path='cora_dyn_graphs.pkl')