from tools.helper import (
    read_pro_go_data,
    read_specific_columns,
    create_ppi_network,
    export_graph_to_pickle
)
from tools.workflow import box_sample_subset
import pandas as pd
import random
import time
from random import sample
from pathlib import Path

input_directory_path = Path("./output/dataset")

go_inferred_columns = [0, 2, 3]
go_protein_pairs = read_pro_go_data(
    "./network/fly_proGo.csv", go_inferred_columns, ["molecular_function", "biological_process", "cellular_component"], ","
)
interactome_columns = [0, 1]
interactome = read_specific_columns("./network/fly_propro.csv", interactome_columns, ",")
protein_list = []

    # if there is no graph.pickle file in the output/dataset directory, uncomment the following lines
graph_file_path = Path(input_directory_path, "graph.pickle")
G, protein_list = create_ppi_network(interactome, go_protein_pairs)
export_graph_to_pickle(G, graph_file_path)
sample_size = 1000
upper = 200
lower = 100

# go_protein_pairs, protein_list = box_sample_subset(go_protein_pairs, protein_list, upper, lower)

# num = 2
name = "_mol_bio_cel"


def reverse_sample_data(go_protein_pairs, sample_size, protein_list, G, input_directory_path, num, name):
    """
    Given a sample size, generate positive nad negative datasets.

    Parameters:

    go_protein_pairs {list} : a list containing the edge between a protein and a go-term e.g. [[protein1, go_term1], [protein2, go_term2], ...]
    sample_size {int} : the size of a positive/negative dataset to be sampled
    protein_list {list} : a list of all proteins in the graph
    G {nx.Graph} : graph that represents the interactome and go term connections
    input_directory_path {Path} : Path to directory of the datasets
    num {int} : Number of positive/negative dataset
    name {str} : shorthand for all namespaces used to generate datasets, adds shorthand to .csv name

    Returns:
    positive_dataset, negative_dataset

    """
    positive_dataset = {"protein": [], "go": []}
    negative_dataset = {"protein": [], "go": []}
    # sample the data
    for edge in sample(list(protein_list), sample_size):
        negative_dataset["protein"].append(edge['id'])

    for protein in negative_dataset['protein']:
        s = random.choice(go_protein_pairs)
        go = s[1]

        while G.has_edge(protein, go):
            s = random.choice(go_protein_pairs)
            go = s[1]
        positive_dataset["protein"].append(s[0])
        positive_dataset["go"].append(go)
        negative_dataset["go"].append(go)

    positive_df = pd.DataFrame(positive_dataset)
    negative_df = pd.DataFrame(negative_dataset)
    
    positive_df.to_csv(
        Path(input_directory_path, "rep_" + str(num) + "_positive_protein_go_term_pairs" + name + ".csv"),
        index=False,
        sep="\t",
    )
    negative_df.to_csv(
        Path(input_directory_path, "rep_" + str(num) + "_negative_protein_go_term_pairs" + name + ".csv"),
        index=False,
        sep="\t",
    )

    return positive_dataset, negative_dataset


for num in range(10):
    reverse_sample_data(go_protein_pairs, sample_size, protein_list, G, input_directory_path, num, name)
    print("Sample " + str(num) + " created")
