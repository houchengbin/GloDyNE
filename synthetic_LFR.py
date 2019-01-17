import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

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

def generate_initial_LFR(n=50):
    tau1 = 3
    tau2 = 1.5
    mu = 0.1
    G = nx.algorithms.community.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=10,seed=4)
    G.remove_edges_from(G.selfloop_edges())
    communities = {frozenset(G.nodes[v]['community']) for v in G}
    degrees = G.degree()
    print(G,degrees)
    community_dict = {}
    communities = list(communities)
    for i in range(len(communities)):
        community = list(communities[i])
        for j in range(len(community)):
            community_dict[str(community[j])] = i
            #print(community[j],"  ",i)
            #print(community_dict[str(community[j])])
    return G,community_dict,degrees

def generate_dynamic_data(initial_G,community_dict,degrees,time_step = 10,initial_edge_porpotation = 0.7):
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
            print("edge decrease",set(graphs[i-1].edges())-set(graphs[i].edges()))
            print("edge increase",set(graphs[i].edges()) - set(graphs[i-1].edges()))
            print("node decrease",set(graphs[i - 1].nodes()) - set(graphs[i].nodes()))
            print("node increase",set(graphs[i].nodes()) - set(graphs[i - 1].nodes()))

        print("graph_size:",len(graphs[i]))

        plt.show(graphs[i])







g1,g2,g3 = generate_initial_LFR()
generate_dynamic_data(g1,g2,g3)
