import networkx as nx

sizes = [75, 75, 300]
probs = [[0.25, 0.05, 0.02],
         [0.05, 0.35, 0.07],
         [0.02, 0.07, 0.40]]
g = nx.stochastic_block_model(sizes, probs, seed=0)
len(g)

H = nx.quotient_graph(g, g.graph['partition'], relabel=True)
for v in H.nodes(data=True):
    print(round(v[1]['density'], 3))




for v in H.edges(data=True):
    print(round(1.0 * v[2]['weight'] / (sizes[v[0]] * sizes[v[1]]), 3))