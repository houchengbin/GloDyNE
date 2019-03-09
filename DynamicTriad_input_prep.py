import networkx as nx
import pickle
import os


def load_nx_graph(path='AS733_dyn_graphs.pkl'):
    with open(path, 'rb') as f:
        nx_graph_reload = pickle.load(f)
    return nx_graph_reload


def get_node_ID_across_graphs(nx_graphs):
    node_lists = []
    for i in range(len(nx_graphs)):
        node_lists.append(nx_graphs[i].nodes)
    node_union = set().union(*node_lists)
    return node_union


def TNE_single_graph(nx_graph, input_name='', graph_count=0):
    if input_name != '':
        output_location = input_name + '_DynamicTriad_prep/'
    else:
        output_location = 'DynamicTriad_prep/'
    file_name = output_location + str(graph_count)
    dirname = os.path.dirname(file_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    f = open(file_name, "w+")
    nodes = list(nx_graph.nodes)
    line = ''
    all_nodes = list(nx_graph.nodes)
    all_nodes.sort()
    for i in range(len(nodes)):
        line += TNE_graph_line(i, nx_graph, all_nodes)

    f.write(line)
    f.close()


def TNE_graph_line(index, nx_graph, all_nodes):
    neighbors = list(nx_graph.neighbors(all_nodes[index]))
    replaced_neighbors = []
    for i in range(len(neighbors)):
        replaced_neighbors.append(all_nodes.index(neighbors[i]))
    replaced_neighbors.sort()
    line = str(index)
    for i in range(len(replaced_neighbors)):
        line += ' ' + str(int(replaced_neighbors[i])) + ' ' + '1.0'
    line += '\n'
    return line


"""
def create_lookup_tables(nx_graphs):
    all_nodes = get_node_ID_across_graphs(nx_graphs)
    all_nodes = [int(x) for x in all_nodes]
    node_lookup_dir_list = []
    node_lookback_list_list = []
    for i in range(len()):
"""


def DynamicTriad_prep(path='AS733_dyn_graphs.pkl', input_name='', add_all_nodes=True):
    print("start finding all nodes")
    nx_graphs = load_nx_graph(path)
    if add_all_nodes == True:
        node_union = get_node_ID_across_graphs(nx_graphs)
        for i in range(len(nx_graphs)):
            subset = list(node_union - nx_graphs[i].nodes)
            for j in range(len(subset)):
                nx_graphs[i].add_node(subset[j])
    print("finish finding all nodes")
    # node_id,number_non-zero:index1,weight1:index2,weight2:...index_d,weightd
    # 0,      2:              1,     1.0:    2,     1.0
    look_up_table = {}
    # nodes = nx_graphs.nodes
    print("start to generating DynamicTriad input")
    for i in range(len(nx_graphs)):
        print("generating input:", i + 1, "out of ", len(nx_graphs))

        TNE_single_graph(nx_graphs[i], graph_count=i)


if __name__ == '__main__':
    DynamicTriad_prep(path='AS733_dyn_graphs.pkl', input_name="AS_733")  # input paht = 'balabala'
    """
    a = [1,2]
    b = [2,3]
    c = [4,5]
    d = [a,b,c]
    print(set().union(*d))
    """
    """

    input_name = ''
    file_name = input_name+'TNEoutput' + str(3) + '.txt'
    print(file_name)
    print('TNEoutput' + str(3) + '.txt')
    """
