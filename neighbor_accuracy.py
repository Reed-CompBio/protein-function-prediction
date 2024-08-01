import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import colormaps
import time
import os
from tools.helper import(
    read_specific_columns,
    read_pro_go_data,
    import_graph_from_pickle,
    get_neighbors,
    print_progress,
)
"""
Iterates through the list of go terms pulled from go_depth.csv. For each go term:
    - Pagerank is calculated for each protein neighbor then ranked by the diffusion score
        -ProGO edge is removed for each protein neighbor then re-added before moving to the next protein neighbor
    - If a go term has n neighbors then when ranked each protein neighbor should be <= nth rank to be accurately predicted
    -The percent of neighbors accurately predicted is caculated (0-100%) 
    
Once that is calculated for each go term, they are then saved in a .csv file and visualized on a scatterplot comparing number of neighbors to the %accurately predicted. 
"""

G = import_graph_from_pickle("./output/dataset/graph.pickle")
# P = import_graph_from_pickle("./output/dataset/protein.pickle")
# D = import_graph_from_pickle("./output/dataset/go_protein.pickle")


# Accepted Deviation, 0 is accurately predicted, 3 is within 3 of accurately predicted, etc... 
x = 0

go_terms = read_specific_columns("./network/go_depth.csv", [0], ",")
go_terms = random.sample(go_terms, 100) #Picks only 100 go_terms to run, comment out line to run all go terms with < 100 neighbors

all_go_neighbor_rank = []
all_go_neighbor_num = []
go_term = []
zero_count = 0
counter = 1

for go in go_terms: 
    go = go[0]
    go_neighbors = get_neighbors(G, go, "protein_go_term")
    num_neighbors = len(go_neighbors)
    if num_neighbors != 0 and num_neighbors <= 100: #Set range using number of go term neighbors (less neighbors = faster)
        print(go)
        print("Number of Neighbors: " + str(num_neighbors))
        print("GO term number " + str(counter))
        counter += 1
        
        pro_dict_all = {}
        pro_dict_progo = {}
        
        t = 1
        total_progress = num_neighbors
        start = time.time()
        #Remove edge between current neighbor and GO term before calculating pagerank
        for i in go_neighbors:
            pro = i[0]
            G.remove_edge(pro, go)
            p = nx.pagerank(G, alpha=0.7, personalization={go:1}) 
            pro_dict_all[pro] = p
            G.add_edge(pro, go, type = "protein_go_term")
            print_progress(t, total_progress)
            t += 1
        print("")
        print(time.time()-start)
        print("\n")
        
        neighbor_ranked = []
        for i in pro_dict_all.keys():
            sort_all = list(sorted(pro_dict_all[i].items(), key=lambda item: item[1], reverse = True))
            protein_dict_all = {}
            count = 1
            #Pull out only proteins (as go terms also have diffusion scores)
            for j in sort_all:
                if j[0][:2] == "FB":
                    protein_dict_all[j[0]] = [j[1], count]
                    count += 1
            neighbor_ranked.append([i, protein_dict_all[i][1]])
            
        #calculate number of times the rank is accurately predicted with a possible deviation of x 
        
        rank_percent = 0
        progo_rank_percent = 0
        l = len(neighbor_ranked)
        limit = l + x
        for i in neighbor_ranked:
            if i[1] <= limit:
                rank_percent += 1
        rank = (rank_percent/l)*100
        
        all_go_neighbor_rank.append(rank)
        all_go_neighbor_num.append(num_neighbors)
        go_term.append(go)
    else:
        zero_count += 1
        all_go_neighbor_rank.append(None)
        all_go_neighbor_num.append(num_neighbors)
        go_term.append(go)

output = []
for i in range(len(go_term)):
    if all_go_neighbor_rank[i] == None:
        output.append([go_term[i], all_go_neighbor_num[i], "N/A"])
    else:
        output.append([go_term[i], all_go_neighbor_num[i], round(all_go_neighbor_rank[i],2)])
df = pd.DataFrame(output, columns = ["GO Term", "Number of Neighbors", "Rank"],)
df.to_csv(
    Path("./output/data/go_neighbor_tests/neighbor_rank", "all_neighbor_rank.csv"),
    index=False,
    sep="\t",
)

print("Number of GO terms with no neighbors: " + str(zero_count))

fig, ax = plt.subplots()
plt.scatter(all_go_neighbor_rank, all_go_neighbor_num)
plt.xlabel("% Go Neighbors Accurately Predicted")
ax.set_xlim([-5, 105])
plt.ylabel("Number of Neighbors")
plt.savefig("./output/data/go_neighbor_tests/neighbor_rank/all_rank.png")
plt.show()
