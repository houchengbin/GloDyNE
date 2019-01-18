import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

"""
n = 250
tau1 = 3
tau2 = 1.5
mu = 0.1
G = nx.algorithms.community.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=7, min_community=30)
communities = {frozenset(G.nodes[v]['community']) for v in G}

print(communities)
print(list(list(list(communities))[0]))
print(G.degree())
nx.draw_networkx(G)
plt.show(G)
"""

def generate_initial_LFR(n=100):
    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    G = nx.algorithms.community.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=10,seed=4)
    G.remove_edges_from(G.selfloop_edges())

    degree_dict = dict(G.degree()) # {node ID: degree, ...}
    # print('degree_dict', degree_dict)


    communities = {frozenset(G.nodes[v]['community']) for v in G}
    community_dict = {}
    communities = list(communities)
    for i in range(len(communities)): # i is community ID
        community = list(communities[i])
        for j in range(len(community)): # j goes over all nodes
            community_dict[community[j]] = i
    # print('community_dict', community_dict) # {node ID: community ID, ...}

    return G, community_dict, degree_dict

def generate_dynamic_data(initial_G, community_dict, degrees,time_step=3, initial_edge_porpotation=0.7):
    reference_graph = initial_G.copy()
    initial_nodes_number = len(degrees)
    initial_edges_number = len(initial_G.edges())
    initial_edges = list(initial_G.edges())
    #disappear_nodes_number = int(initial_edges_number*initial_edge_porpotation)
    chance_each_time_step = (1-initial_edge_porpotation)/time_step

    time_step_slots = []
    time_step_slots.append(initial_edge_porpotation)
    for i in range(time_step):
        time_step_slots.append(initial_edge_porpotation + (i+1)*chance_each_time_step)


    edges_time_step = []
    for i in range(initial_edges_number):
        random_number = random.random()
        for j in range(len(time_step_slots)):
            if time_step_slots[j] > random_number:
                edges_time_step.append(j)
                break

    graphs = []
    for i in range(time_step+1):
        graphs.append(nx.Graph())
    for i in range(len(edges_time_step)):
        current_edge_time_step = edges_time_step[i]
        for j in range(current_edge_time_step, len(graphs)):
            graphs[j].add_edge(list(initial_edges[i])[0],list(initial_edges[i])[1])


    for i in range(len(graphs)):
        nx.draw_networkx(graphs[i])
        if i > 0:
            print("edge deleted",set(graphs[i-1].edges()) - set(graphs[i].edges()))
            print("edge added",set(graphs[i].edges()) - set(graphs[i-1].edges()))
            print("node deleted",set(graphs[i - 1].nodes()) - set(graphs[i].nodes()))
            print("node added",set(graphs[i].nodes()) - set(graphs[i - 1].nodes()))
            # 为了解决这个问题，这里需要再看看LRF生成的图是否应该有方向。如果没有方向，直接把每个graph变为undirect就行了
            # https://networkx.github.io/documentation/stable/reference/functions.html#graph
            # G = nx.to_undirected(G)
            # 另外，在增加减少边时候，还要注意一个edge (a,b) 和 (b,a)的问题，可能需要一个小函数来处理

            # 如果是有方向的，我们还需要进一步考虑：1）我们只针对无向图做吗？2）如果也做有向图，会遇到什么大麻烦吗？
            # 我个人想法，我还还是focus在无方向图，具体原因我们在讨论。

        print("graph_size: ", len(graphs[i]), '====== @ time step', i)

        plt.show(graphs[i])

    return graphs



def save_nx_graph(nx_graph, path='nx_graph_temp.data'):
    with open(path, 'wb') as f:
        pickle.dump(nx_graph, f, protocol=pickle.HIGHEST_PROTOCOL) #the higher protocol, the smaller file
    with open(path, 'rb') as f:
        nx_graph_reload = pickle.load(f)
    
    try:
        print('Check if it is correctly dumped and loaded: ', nx_graph_reload.edges() == nx_graph.edges(), ' It contains only ONE graph')
    except:
        for i in range(len(nx_graph)):
            print('Check if it is correctly dumped and loaded: ', nx_graph_reload[i].edges() == nx_graph[i].edges(), ' for Graph ', i)


def save_any_obj(obj, path='obj_temp.data'):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':

    G, community_dict, degree_dict = generate_initial_LFR()
    save_nx_graph(nx_graph=G, path='LFR_static_graph.data')
    save_any_obj(obj=community_dict, path='LFR_community_dict.data')  # {node ID: degree, ...}
    save_any_obj(obj=degree_dict, path='LFR_degree_dict.data')  # {node ID: community ID, ...}

    Gs = generate_dynamic_data(G, community_dict, degree_dict)
    save_nx_graph(nx_graph=Gs, path='LFR_dynamic_graphs.data')