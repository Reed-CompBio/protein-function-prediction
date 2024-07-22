from tools.helper import (
    read_pro_go_data,
    read_specific_columns,
    create_ppi_network,
    export_graph_to_pickle,
    print_progress,
)
from tools.workflow import remove_samples
import pandas as pd
import random
import time
from random import sample
from pathlib import Path

input_directory_path = Path("./output/dataset")

namespace = ["molecular_function", "biological_process", "cellular_component"]
    # change the go_term_type variable to include which go term namespace you want
go_term_type = [namespace[0]]
short_name = ""
if namespace[0] in go_term_type:
    short_name = short_name + "_mol"
if namespace[1] in go_term_type:
    short_name = short_name + "_bio"
if namespace[2] in go_term_type:
    short_name = short_name + "_cel"

go_inferred_columns = [0, 2, 3]
go_protein_pairs = read_pro_go_data(
    "./network/fly_proGo.csv", go_inferred_columns, go_term_type, ","
)

interactome_columns = [0, 1]
interactome = read_specific_columns("./network/fly_propro.csv", interactome_columns, ",")
protein_list = []

    # if there is no graph.pickle file in the output/dataset directory, uncomment the following lines
graph_file_path = Path(input_directory_path, "graph.pickle")
G, protein_list = create_ppi_network(interactome, go_protein_pairs)
export_graph_to_pickle(G, graph_file_path)
sample_size = 100
pair_type = "protein_go"
reps = 1
#Options for pair_type: "protein_go", "protein_protein", "both"


def paired_sample_data(go_protein_pairs, sample_size, protein_list, proteins, G, input_directory_path, num, name):
    """
    Given a sample size, generate positive nad negative datasets.

    Parameters:

    go_protein_pairs {list} : a list containing the edge between a protein and a go-term e.g. [[protein1, go_term1], [protein2, go_term2], ...]
    sample_size {int} : the size of a positive/negative dataset to be sampled
    protein_list {list} : a list of all proteins in the graph
    proteins {dict} : a list of all proteins in the graph and the number of annotated go terms
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
    for edge in sample(list(go_protein_pairs), sample_size):
        positive_dataset["protein"].append(edge[0])
        positive_dataset["go"].append(edge[1])

    i = 1

    for protein, go in zip(positive_dataset["protein"], positive_dataset["go"]):
        freq = proteins[protein]
        # print(go_num)
        sample_edge = random.choice(protein_list)
        sample_edge = sample_edge['id']
        r = 10
        t = .1
        start = time.time()
        # removes if a protein has a corresponding edge to the GO term in the network
        while G.has_edge(sample_edge, go) or not ((freq-r) <= proteins[sample_edge] <= (freq+r)):
            sample_edge = random.choice(protein_list)
            sample_edge = sample_edge['id']
            end = time.time()
            if (end-start) > t: #Slightly arbitrary, but the program cycles through all options relatively fast
                r += 10
                t += .01
        negative_dataset["protein"].append(sample_edge)
        negative_dataset["go"].append(go)
        print_progress(i, sample_size)
        i += 1

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


def paired_sample_data_multi_input(go_protein_pairs, sample_size, protein_list, proteins, G, input_directory_path, num, name):
    """
    Given a sample size, generate positive nad negative datasets.

    Parameters:

    go_protein_pairs {list} : a list containing the edge between a protein and a go-term e.g. [[protein1, go_term1], [protein2, go_term2], ...]
    sample_size {int} : the size of a positive/negative dataset to be sampled
    protein_list {list} : a list of all proteins in the graph
    proteins {dict} : a list of all proteins in the graph and the number protein neighbors and annotated go terms
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
    for edge in sample(list(go_protein_pairs), sample_size):
        positive_dataset["protein"].append(edge[0])
        positive_dataset["go"].append(edge[1])

    i = 1

    for protein, go in zip(positive_dataset["protein"], positive_dataset["go"]):
        freq_pro = proteins[protein][0]
        freq_go = proteins[protein][1]
        sample_edge = random.choice(protein_list)
        sample_edge = sample_edge['id']
        r = 10
        t = .01
        start = time.time()
        # removes if a protein has a corresponding edge to the GO term in the network
        while G.has_edge(sample_edge, go) or not ((freq_pro-r) <= proteins[sample_edge][0] <= (freq_pro+r)) or not ((freq_go-r) <= proteins[sample_edge][1] <= (freq_go+r)):
            sample_edge = random.choice(protein_list)
            sample_edge = sample_edge['id']
            end = time.time()
            if (end-start) > t: #Slightly arbitrary, but the program cycles through all options relatively fast
                r += 10
                t += .01
        negative_dataset["protein"].append(sample_edge)
        negative_dataset["go"].append(go)
        print_progress(i, sample_size)
        i += 1

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
    



def go_neighbor_sample_list(proGo, proLst): 
    """
    Creates a list of how many go term neighbors each protein has

    Parameters:

    proGo {list}: a list of protein go term pairs 
    proLst {list} : a list of protein nodes
    
    Returns:
    a dictionary containing the number of go term neighbors for all proteins

    """
    proteins = {} # Number of proteins annotated to each go term
    for term in proLst:
        x = term['id']
        if x not in proteins.keys():
            proteins[x] = 0
            
    for term in proGo:
        i = term[0]
        proteins[i] += 1

    df = pd.DataFrame.from_dict(proteins, orient = "index")
    df.to_csv(
        Path("./output/data", "number_of_annotated_go_terms.csv"),
        index=True,
        sep="\t",
    )

    return proteins
    
def protein_neighbor_sample_list(proproLst): 
    """
    Creates a list of how many go term neighbors each protein has

    Parameters:
    proproLst {list} : a list protein neighbors
    
    Returns:
    a dictionary containing the number of protein neighbors for all proteins

    """
    proteins = {} # Number of proteins neighbors for each protein
    for term in proproLst:
        x = term[0]
        y = term[1]
        if x not in proteins.keys():
            proteins[x] = 1
        else: 
            proteins[x] += 1
            
        if y not in proteins.keys():
            proteins[y] = 1
        else:
            proteins[y] += 1

    df = pd.DataFrame.from_dict(proteins, orient = "index")
    df.to_csv(
        Path("./output/data", "number_of_annotated_proteins.csv"),
        index=True,
        sep="\t",
    )

    return proteins

remove_samples("./output/dataset")
if pair_type == "protein_protein":
    proteins = protein_neighbor_sample_list(interactome)
elif pair_type == "protein_go":
    proteins = go_neighbor_sample_list(go_protein_pairs, protein_list) #Make this a perminant list?
elif pair_type == "both":
    proteins = protein_neighbor_sample_list(interactome)
    protein_go = go_neighbor_sample_list(go_protein_pairs, protein_list)
    for i in proteins:
        proteins[i] = [proteins[i], 0] #Protein neighbors in index 0, go neighbors in index 1
    for i in protein_go:
        proteins[i][1] = protein_go[i] 
    for num in range(reps):
        paired_sample_data_multi_input(go_protein_pairs, sample_size, protein_list, proteins, G, input_directory_path, num, short_name)
        print(" Sample " + str(num) + " created")
        
if pair_type != "both":
    for num in range(reps):
        paired_sample_data(go_protein_pairs, sample_size, protein_list, proteins, G, input_directory_path, num, short_name)
        print(" Sample " + str(num) + " created")