import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle

def read_youtube_data():#read graph
    #communities = np.genfromtxt("com-youtube.all.cmty.txt", dtype=str)
    graph_data = np.genfromtxt("com-youtube.ungraph.txt", dtype=str)
    community_dict = {}
    return graph_data


def generate_youtube_community(community_dict, top_5000 = False):
    if top_5000 == False:
        text_file = open("com-youtube.all.cmty.txt", "r")
    else:
        text_file = open("com-youtube.top5000.cmty.txt", "r")
    lines = text_file.read().split('\n')
    text_file.close()

    for i in range(len(lines) - 1):
        text_split = lines[i].split('\t')
        for j in range(len(text_split)):
            node_community = community_dict[text_split[j]]
            print("before",community_dict[text_split[j]])
            node_community.append(str(i))
            community_dict.update({text_split[j]: node_community})
            print("after:",community_dict[text_split[j]])
    return community_dict
def generate_initial_graph(top_5000 = False):
    initial_graph = nx.Graph()
    graph_data = read_youtube_data()

    for i in range(len(graph_data)):
        print("adding edges",i," out of ",len(graph_data))
        initial_graph.add_edge(graph_data[i][0],graph_data[i][1])
    community_dict = {}
    all_nodes = list(initial_graph.nodes())
    for i in range(len(all_nodes)):
        community_dict[all_nodes[i]] = []

    community_dict = generate_youtube_community(community_dict)

    count_no_community = 0
    for i in range(len(all_nodes)):
        if len(community_dict[all_nodes[i]]) == 0:
            node_community = [str(-1)]
            community_dict.update({all_nodes[i]: node_community})

    for i in range(len(all_nodes)):
        if len(community_dict[all_nodes[i]]) > 2:
            print(all_nodes[i], " ",community_dict[all_nodes[i]])


    return initial_graph, community_dict


def generate_dynamic_data(initial_G, time_step=5,
                          initial_edge_porpotation=0.5):  # why this name initial_edge_porpotation
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
    """
    for i in range(len(graphs)):
        #nx.draw_networkx(graphs[i])
        if i > 0:
            # print('nx.is_directed(G)?', nx.is_directed(graphs[i]))
            print("edge deleted v1", set(graphs[i - 1].edges()) - set(graphs[i].edges()))
            print("edge deleted v2", edge_s1_minus_s0(graphs[i-1],graphs[i]))
            print("edge added v1", set(graphs[i].edges()) - set(graphs[i - 1].edges()))
            print("edge added v2", edge_s1_minus_s0(graphs[i],graphs[i-1]))
            print("node deleted", set(graphs[i].nodes()) - set(graphs[i-1].nodes()))
            print("node added", set(graphs[i].nodes()) - set(graphs[i - 1].nodes()))
            # 我们这次文章只做undirected graph；
            # 1）直接用networkx的undirected graph，目前就是
            # 2）用networkx的directed graph，但是我们所有的处理都必须是double edge比如只要有(a,b)，必须加或者减（a,b）+(b,a)
            # G_directed = nx.to_directed(G_undirected)

        print("graph_size: ", len(graphs[i]), '====== @ time step', i)

        #plt.show(graphs[i])
        """
    return graphs

def edge_s1_minus_s0(s1, s0, is_directed=False):
    if not is_directed:
        s1_reordered = set((a, b) if int(a) < int(b) else (b, a) for a, b in s1.edges())
        s0_reordered = set((a, b) if int(a) < int(b) else (b, a) for a, b in s0.edges())
        return s1_reordered - s0_reordered
    else:
        print('currently not support directed case')

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
    graph, community = generate_initial_graph()

    print(list(nx.isolates(graph)))

    #graphs = generate_dynamic_data(graph,time_step=4,initial_edge_porpotation=0.6)

    #read_youtube_data(top_5000= False)
    """
    save_any_obj(obj=community, path='youtube_node_label_dict.pkl')  # {node ID: degree, ...}
    save_nx_graph(nx_graph=graphs, path='youtube_dyn_graphs.pkl')
    """