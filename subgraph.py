import random
import networkx as nx
import matplotlib.pyplot as plt
from tools.helper import (
    import_graph_from_pickle,
    get_neighbors,
    export_graph_to_pickle,
    read_specific_columns,
)

'''
Creates a subgraph using the top pageranked nodes for a given go term that can visualize the paths between nodes. This can be done with four different graphs, one with inferred edges (G), one without inferred edges (I), one with only protein-protein edges (P), and one with protein-go term edges only. 

'''

go_term = "GO:0015140"

#Uses pre-generated graphs from main (with the exception of I, which is renamed)
G = import_graph_from_pickle("./output/dataset/graph.pickle")
I = import_graph_from_pickle("./output/dataset/no_inferred.pickle")
P = import_graph_from_pickle("./output/dataset/protein.pickle")
D = import_graph_from_pickle("./output/dataset/go_protein.pickle")

#Which graphs are used for the go term, each graph has a unique output
graphs = [G, I, P, D]
# Number of top ranked nodes to visualize 
num = 30
    
for graph in graphs:
    go_neighbors = get_neighbors(G, go_term, "protein_go_term")
    n = []
    for i in go_neighbors:
        n.append(i[0])

    # Randomly pickls one of the Go term's protein neighbors as the positive protein 
    neighbor = random.sample(n, 1)
    neighbor = neighbor[0]
    n.remove(neighbor)

    t = "" 
    if graph == G:
        t = "G"
    elif graph == I:
        t = "I"
    elif graph == P:
        t = "P"
    elif graph == D:
        t = "D"
    
    # Removes the edge to the positive protein and runs pagerank
    if t == "P":
        pro = {}
        for j in n:
            pro[j] = 1
        p = nx.pagerank(graph, alpha=0.7, personalization=pro) 
    else:
        graph.remove_edge(neighbor, go_term)
        p = nx.pagerank(graph, alpha=0.7, personalization={go_term:1}) 
        
        
    sort_all = list(sorted(p.items(), key=lambda item: item[1], reverse = True))

    # Pulls only the top num ranked neighbors (protein and go term)
    node_lst = []
    count = 1
    for i in sort_all:
        if count <= num:
            node_lst.append(i[0])
        if i[0] == neighbor:
            print(i[0] + " is ranked " + str(count))
        count += 1

    # Creates a subgraph using the top num ranked nodes 
    S = graph.subgraph(node_lst)
    

    # Colorcodes the nodes, red for the go term, dark green for the positive protein, pink for the go term neighbors and teal for everything else 
    color = []
    Ecolor = []
    for i in S.nodes():
        if i == go_term:
            color.append("firebrick")
        elif i == neighbor:
            color.append("darkolivegreen")
        elif i in n:
            color.append("mediumvioletred")
        else:
            color.append("mediumturquoise")

    # Colorcodes the edges, dark yellow for the go term neighbors, pale orange for the positive protein neighbors, and purple for the neighbors of the go term annotated proteins, black for all else
    for i in S.edges():
        if i[1] == go_term or i[0] == go_term:
            Ecolor.append("darkgoldenrod")
        elif i[1] == neighbor or i[0] == neighbor:
            Ecolor.append("peachpuff")
        elif i[0] in n or i[1] in n:
            Ecolor.append("rebeccapurple")
        else:
            Ecolor.append("black")

    titles = {"G": "With Inferred Edges",
              "I": "No Inferred Edges",
              "P": "Protein-protein Edges Only",
              "D": "ProGo Edges Only"
    }
    
    pos = nx.spring_layout(S, seed=63)
    fig, ax = plt.subplots()
    plt.title(titles[t])
    nx.draw_networkx(S, pos, with_labels = True, node_color = color, edge_color = Ecolor)
    plt.show() 
