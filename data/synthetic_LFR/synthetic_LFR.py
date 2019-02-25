import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle


'''
The final graph follow LFR distribution? We randomly preserve 40% edges and then equally add edges for 15 steps ...

@ graph 0 # of nodes 4313 # of edges 4762
@ graph 1 # of nodes 4465 # of edges 5216
@ graph 2 # of nodes 4581 # of edges 5653
@ graph 3 # of nodes 4692 # of edges 6105
@ graph 4 # of nodes 4792 # of edges 6565
@ graph 5 # of nodes 4860 # of edges 7049
@ graph 6 # of nodes 4907 # of edges 7457
@ graph 7 # of nodes 4946 # of edges 7887
@ graph 8 # of nodes 4991 # of edges 8326
@ graph 9 # of nodes 5024 # of edges 8741
@ graph 10 # of nodes 5059 # of edges 9164
@ graph 11 # of nodes 5081 # of edges 9586
@ graph 12 # of nodes 5091 # of edges 9995
@ graph 13 # of nodes 5098 # of edges 10404
@ graph 14 # of nodes 5099 # of edges 10830
'''

# -----------------------------------------------------------------
# ------------------------- data generator ------------------------
# -----------------------------------------------------------------
def generate_initial_LFR(n=5100, tau1=3, tau2=1.5, mu=0.1, average_degree=4, min_community=30, 
                            max_community=None, seed=0):
    # https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.community_generators.LFR_benchmark_graph.html#networkx.algorithms.community.community_generators.LFR_benchmark_graph
    G = nx.algorithms.community.LFR_benchmark_graph(n=n, tau1=tau1, tau2=tau2, mu=mu,
                                                    average_degree=average_degree, min_community=min_community,
                                                    max_community=max_community, seed=seed)


    G.remove_edges_from(G.selfloop_edges())
    print('The initial LFR graph is nx.is_directed(G)?', nx.is_directed(G))

    isolated_nodes = list(nx.isolates(G)).copy()
    print(isolated_nodes)

    degree_dict = dict(G.degree())  # {node ID: degree, ...}
    # print('degree_dict', degree_dict)

    communities = {frozenset(G.nodes[v]['community']) for v in G}
    community_dict = {}
    communities = list(communities)
    for i in range(len(communities)):  # i is community ID
        community = list(communities[i])
        for j in range(len(community)):  # j goes over all nodes
            community_dict[str(community[j])] = i

    for i in range(len(isolated_nodes)):
        del community_dict[str(isolated_nodes[i])]

    print(len(community_dict), " community size")
    print(len(set(community_dict.values())), "number of community")
    # print('community_dict', community_dict)  # {node ID: community ID, ...}
    return G, community_dict, degree_dict


def generate_dynamic_data(initial_G, time_step=15, initial_edge_perc=0.4):  # why this name initial_edge_perc
    # reference_graph = initial_G.copy()
    initial_nodes_number = len(initial_G.nodes())
    initial_edges_number = len(initial_G.edges())
    initial_edges = list(initial_G.edges())
    # disappear_nodes_number = int(initial_edges_number*initial_edge_perc)
    chance_each_time_step = (1 - initial_edge_perc) / time_step

    time_step_slots = []
    time_step_slots.append(initial_edge_perc)
    for i in range(time_step):
        time_step_slots.append(initial_edge_perc + (i + 1) * chance_each_time_step)

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

def save_nx_graph(nx_graph, path='nx_graph_temp.pkl'):
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


def save_any_obj(obj, path='obj_temp.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

# --------------------------------------------------
# -------------------- test  -----------------------
# --------------------------------------------------
if __name__ == '__main__':
    G, community_dict, degree_dict = generate_initial_LFR()
    # save_nx_graph(nx_graph=G, path='LFR_static_graph.pkl')
    save_any_obj(obj=community_dict, path='LFR_node_label_dict.pkl')  # {node ID: community/label ID, ...} 
    # save_any_obj(obj=degree_dict, path='LFR_degree_dict.pkl')  # {node ID: degree, ...}

    graphs = generate_dynamic_data(G)
    print(len(graphs[-1].nodes()))
    save_nx_graph(nx_graph=graphs[-15:], path='LFR_dyn_graphs.pkl')

    graphs = graphs[-15:]
    for i in range(len(graphs)):
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))