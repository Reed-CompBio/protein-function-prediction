import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import colormaps
from tools.helper import(
    read_specific_columns,
    read_pro_go_data,
    import_graph_from_pickle,
    get_neighbors,
)

G = import_graph_from_pickle("./output/dataset/graph.pickle")
# P = import_graph_from_pickle("./output/dataset/protein.pickle")
D = import_graph_from_pickle("./output/dataset/go_protein.pickle")

# namespace = ["molecular_function", "biological_process", "cellular_component"]
# # change the go_term_type variable to include which go term namespace you want
# go_term_type = [namespace[0], namespace[1], namespace[2]]

# go_inferred_columns = [0, 2, 3]
# go_protein_pairs = read_pro_go_data(
#     "./network/fly_proGo.csv", go_inferred_columns, go_term_type, ","
# )

# go_terms = read_specific_columns("./network/go_term.csv", [0], ",")
# flag = 0
# while flag <= 1:
#     go = random.choice(go_terms)
#     go = go[0]
#     go_neighbors = get_neighbors(D, go, "protein_go_term")
#     flag = len(go_neighbors)

go = "GO:0065007"
go_neighbors = get_neighbors(D, go, "protein_go_term")
num_neighbors = len(go_neighbors)
print(go)
print("Number of Neighbors: " + str(num_neighbors))

pro_dict_all = {}
pro_dict_progo = {}

for i in go_neighbors:
    pro = i[0]
    G.remove_edge(pro, go)
    p = nx.pagerank(G, alpha=0.7, personalization={go:1}) 
    pro_dict_all[pro] = p
    G.add_edge(pro, go, type = "protein_go_term")

for i in go_neighbors:
    pro = i[0]
    D.remove_edge(pro, go)
    p = nx.pagerank(D, alpha=0.7, personalization={go:1}) 
    pro_dict_progo[pro] = p
    D.add_edge(pro, go, type = "protein_go_term")

neighbor_ranked = []
for i in pro_dict_all.keys():
    sort_all = list(sorted(pro_dict_all[i].items(), key=lambda item: item[1], reverse = True))
    sort_progo = list(sorted(pro_dict_progo[i].items(), key=lambda item: item[1], reverse = True))
    protein_dict_all = {}
    protein_dict_progo = {}
    count = 1
    for j in sort_all:
        if j[0][:2] == "FB":
            protein_dict_all[j[0]] = [j[1], count]
            count += 1
    count = 1
    for j in sort_progo:
        if j[0][:2] == "FB":
            protein_dict_progo[j[0]] = [j[1], count]
            count += 1
    neighbor_ranked.append([i, protein_dict_all[i][1], protein_dict_progo[i][1]])
    print("Protein: " + i + "  Rank: " + str(protein_dict_all[i][1]) + "  ProGo Rank: " + str(protein_dict_progo[i][1]))
    
print(neighbor_ranked)


df = pd.DataFrame(neighbor_ranked, columns = ["Protein", "Rank", "ProGo Only Rank"],)
df.to_csv(
    Path("./output/data/go_neighbor_tests/neighbor_rank", go + "_ranked_neighbors.csv"),
    index=False,
    sep="\t",
)

# #Creates a small network output
# S = nx.Graph()
# size = []
# color = []
# S.add_node(go)
# nx.set_node_attributes(S, go, p[go])
# size.append(600)
# color.append(p[go])

# vmin = 0
# vmax = 1
# for i in go_neighbors:
#     i = i[0]
#     S.add_node(i)
#     S.add_edge(i, go)
#     x = p[i]
#     nx.set_node_attributes(S, i, x)
#     size.append(400)
#     color.append(x)
#     if p[i] < vmin:
#         vmin = p[i]
#     if p[i] > vmax:
#         vmax = p[i]


# prev = []
# for i in go_neighbors:
#     i = i[0]
#     for j in prev:
#         if G.has_edge(i, j):
#             S.add_edge(i, j)
#     prev.append(i)

# for i in go_neighbors:
#     i = i[0]
#     protein_neighbors = get_neighbors(G, i, "protein_protein")
#     for j in protein_neighbors:
#         j = j[0]
#         if j not in S.nodes():
#             S.add_node(j)
#             x = p[j]
#             nx.set_node_attributes(S, j, x)
#             size.append(300)
#             color.append(x)
#             if p[j] < vmin:
#                 vmin = p[j]
#             if p[j] > vmax:
#                 vmax = p[j]
#         S.add_edge(j, i)
#     gopro_neighbors = get_neighbors(G, i, "protein_go_term")
#     for j in gopro_neighbors:
#         j = j[0]
#         if j not in S.nodes():
#             S.add_node(j)
#             x = p[j]
#             nx.set_node_attributes(S, j, x)
#             size.append(300)
#             color.append(x)
#             if p[j] < vmin:
#                 vmin = p[j]
#             if p[j] > vmax:
#                 vmax = p[j]
#         S.add_edge(j, i)
        

# fig, ax = plt.subplots()
# nx.draw_networkx(S, with_labels = True, node_size = size, node_color = color, cmap = plt.cm.cool, font_size = 8)

# cbar = plt.colorbar(
#     plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin, vmax)),
#     cax=ax.inset_axes([0.95, 0.1, 0.05, 0.8]),
# )

# plt.show()
# # for i in go_neighbors:
    


# dict_order = {}
# for i in pro_dict.keys():
#     order = [["",0],["",0],["",0],["",0],["",0],["",0],["",0],["",0],["",0],["",0]]
#     for j in pro_dict[i].keys():
#         if j[:2] != "GO":
#             val = pro_dict[i][j]
#             if val > order[0][1]:
#                 index = 0
#             elif val > order[1][1]:
#                 index = 1
#             elif val > order[2][1]:
#                 index = 2
#             elif val > order[3][1]:
#                 index = 3
#             elif val > order[4][1]:
#                 index = 4
#             elif val > order[5][1]:
#                 index = 5
#             elif val > order[6][1]:
#                 index = 6
#             elif val > order[7][1]:
#                 index = 7
#             elif val > order[8][1]:
#                 index = 8
#             elif val > order[9][1]:
#                 index = 9  
#             else:
#                 index = 10
#             order.insert(index,[j, val])
#             order.pop()
#     dict_order[i] = order


# df = pd.DataFrame(dict_order)
# df.to_csv(
#     Path("./output/data/go_neighbor_tests/top_ten", go + "_top_ten_scored_proteins.csv"),
#     index=False,
#     sep="\t",
# )

#Find a way to visualize a sub-network with the top 10 values and any intermediate connections to see how diffusion looks for a single go term.