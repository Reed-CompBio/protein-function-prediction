from tools.helper import (
    read_pro_go_data,
    read_specific_columns,
    create_ppi_network,
    export_graph_to_pickle,
    print_progress,
)
import networkx as nx
# import pandas as pd
# import random
# import time
from pathlib import Path

input_directory_path = Path("./output/dataset")

go_inferred_columns = [0, 2, 3]
go_protein_pairs = read_pro_go_data(
    "./network/fly_proGo.csv", go_inferred_columns, ["molecular_function", "biological_process", "cellular_component"], ","
)

interactome_columns = [0, 1]
interactome = read_specific_columns("./network/fly_propro.csv", interactome_columns, ",")
protein_list = []

graph_file_path = Path(input_directory_path, "graph.pickle")
G, protein_list = create_ppi_network(interactome, go_protein_pairs)
export_graph_to_pickle(G, graph_file_path)

#Positive: FBgn0004855	GO:0006950
#Negative: FBgn0020638	GO:0006950
if not G.has_edge("FBgn0004855", "GO:0006950"):
    print("Fuck")
G.remove_edge("FBgn0004855", "GO:0006950")
if not G.has_edge("FBgn0004855", "GO:0006950"):
    print("Removed edge")
p = nx.pagerank(G, alpha=0.85, personalization={'GO:0006950':1})
print(p['FBgn0004855'])
print(p['FBgn0020638'])
G.add_edge("FBgn0004855", "GO:0006950")
if G.has_edge("FBgn0004855", "GO:0006950"):
    print("Successfully readded edge")
