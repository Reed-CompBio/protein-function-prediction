from tools.helper import (
    read_pro_go_data,
    read_specific_columns,
    create_ppi_network,
    export_graph_to_pickle,
    print_progress,
)
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
from pathlib import Path

'''
Generates a test graph to visualize how changing aspects of the graph affect pagerank with possibilities of both directed and undirected graphs.

'''

interactome_columns = [0, 1]
interactome = read_specific_columns("./network/fly_propro.csv", interactome_columns, ",")

go_inferred_columns = [0, 2, 3]
go_protein_pairs = read_pro_go_data(
    "./network/fly_proGo.csv", go_inferred_columns, ["molecular_function", "biological_process", "cellular_component"], ","
)

# Test dataset to see the difference between a directed graph and an undirected graph with pagerank
T = nx.DiGraph()
# Undirected 
T.add_node("A")
T.add_node("B")
T.add_node("C")
T.add_node("D")
T.add_node("E")
T.add_node("F")

# T.add_edge("A", "B")
# T.add_edge("B", "C")
# T.add_edge("B", "D")
# T.add_edge("C", "D")
# T.add_edge("C", "E")
# T.add_edge("D", "E")

# Directed
# ["C", "D"] <- Edge removed between "Go term" and "positive protein"
c = [["C", "B"], ["C", "E"], ["A", "B"], ["B", "A"], ["D", "E"], ["E", "D"], ["B", "D"], ["D", "B"], ["F", "A"], ["A", "F"], ["E", "F"], ["F", "E"]]
T.add_edges_from(c)

t = nx.pagerank(T, personalization = {"C" : 1})
lst = []
# Creates range for color coding
vmin = t["A"]
vmax = t["A"]
for i in t:
    x = t[i]
    nx.set_node_attributes(T, i, x)
    lst.append(x)
    if t[i] < vmin:
        vmin = t[i]
    if t[i] > vmax:
        vmax = t[i]
    print(i)
    print(x)

fig, ax = plt.subplots()
nx.draw_networkx(T, with_labels = True, node_size = [300, 300, 500, 300, 300, 300], node_color = lst, cmap = plt.cm.cool)

cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin, vmax)),
    cax=ax.inset_axes([0.95, 0.1, 0.05, 0.8]),
)
plt.savefig(Path("./output/images", "small_graph_with_pagerank_go_both_directed.png"))
plt.show() 