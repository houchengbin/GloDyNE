"""
Code	PH
Category	⬤ Coauthorship
Data source	http://snap.stanford.edu/data/cit-HepPh.html
Vertex type	Author
Edge type	Collaboration
Format	Undirected: Edges are undirected Undirected
Edge weights	Multiple unweighted: Multiple edges are possible Multiple unweighted
Metadata	Timestamps:  Edges are annotated with a timestamps Timestamps
Size	28,093 vertices (authors)
Volume	4,596,803 edges (collaborations)
Unique volume	3,148,447 edges (collaborations)

We equally divide it into 100 time steps and then choose 50-64 time steps

@ graph 0 # of nodes 8063 # of edges 326080
@ graph 1 # of nodes 8331 # of edges 349321
@ graph 2 # of nodes 8682 # of edges 369787
@ graph 3 # of nodes 9039 # of edges 398234
@ graph 4 # of nodes 9317 # of edges 425758
@ graph 5 # of nodes 9590 # of edges 445088
@ graph 6 # of nodes 9905 # of edges 469080
@ graph 7 # of nodes 10321 # of edges 524170
@ graph 8 # of nodes 10534 # of edges 550938
@ graph 9 # of nodes 10846 # of edges 585503
@ graph 10 # of nodes 11092 # of edges 605842
@ graph 11 # of nodes 11397 # of edges 619930
@ graph 12 # of nodes 11768 # of edges 656389
@ graph 13 # of nodes 12105 # of edges 684477
@ graph 14 # of nodes 12474 # of edges 709070
"""

import numpy as np
import networkx as nx
import pickle


def read_txt_file():
    # read the txt file
    lines = np.genfromtxt("out.ca-cit-HepPh.txt", dtype=int)
    return lines


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


def generate_dynamic_graph(time_step_number=100):
    lines = read_txt_file()
    """
    min_mum_non_zero_time = 1015891201
    max_mum_time = 0

    for i in range(len(lines)):
        time_tag = lines[i][3]
        if time_tag > 0:
            if time_tag < min_mum_non_zero_time:
                min_mum_non_zero_time = time_tag
            if time_tag > max_mum_time:
                max_mum_time = time_tag

    print(min_mum_non_zero_time)
    print(max_mum_time)
    """
    min_time = float(700617600)
    max_time = float(1015891201)

    diff = (max_time - min_time) / float(time_step_number)
    time_slot = []

    for i in range(time_step_number - 1):
        time_slot.append(min_time + diff * (1 + i))
    time_slot.append(max_time + 1)
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
            graphs[j].add_edge(str(lines[i][0]), str(lines[i][1]))

    for i in range(len(graphs)):
        print(len(graphs[i].edges), " graph:", i)
        print(len(graphs[i].nodes), " nodes:", i)

    return graphs


if __name__ == '__main__':
    graphs = generate_dynamic_graph()
    save_nx_graph(nx_graph=graphs[50:65], path='hepph_dynamic_graphs.data')

    graphs = graphs[50:65]
    for i in range(len(graphs)):
        print('@ graph', i, '# of nodes', len(graphs[i].nodes()), '# of edges', len(graphs[i].edges()))