from classes.overlapping_neighbors_class import OverlappingNeighbors
from classes.overlapping_neighbors_v2_class import OverlappingNeighborsV2
from classes.overlapping_neighbors_v3_class import OverlappingNeighborsV3
from classes.protein_degree_class import ProteinDegree
from classes.protein_degree_v2_class import ProteinDegreeV2
from classes.protein_degree_v3_class import ProteinDegreeV3
from classes.sample_algorithm import SampleAlgorithm
from classes.hypergeometric_distribution_class import HypergeometricDistribution
from classes.hypergeometric_distribution_class_V2 import HypergeometricDistributionV2
from classes.random_walk_class import RandomWalk
from classes.random_walk_class_v2 import RandomWalkV2
from classes.random_walk_class_v3 import RandomWalkV3
from classes.random_walk_class_v4 import RandomWalkV4
from classes.random_walk_class_v5 import RandomWalkV5

import matplotlib.pyplot as plt
from random import sample
from pathlib import Path
import os
import sys
import pandas as pd
import statistics as stat
from colorama import init as colorama_init
from tools.helper import (
    create_mixed_network,
    create_only_protein_network,
    create_go_protein_only_network,
    read_specific_columns,
    export_graph_to_pickle,
    read_pro_go_data,
    read_go_depth_data
)
from tools.workflow import run_workflow
from networkx.algorithms import bipartite


def main():
    colorama_init()
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("output/dataset"):
        os.makedirs("output/dataset")
    if not os.path.exists("output/data"):
        os.makedirs("output/data")
    if not os.path.exists("output/images"):
        os.makedirs("output/images")

    fly_interactome_path = Path("./network/fly_proPro.csv")
    fly_reg_path = Path("./network/fly_reg.csv")
    fly_go_association_path = Path("./network/fly_proGo.csv")
    fly_go_association_mixed_path = Path("./network/fly_proGo_mixed.csv")
    zfish_interactome_path = Path("./network/zfish_proPro.csv")
    zfish_reg_path = Path("./network/zfish_reg.csv")
    zfish_go_association_path = Path("./network/zfish_proGo.csv")
    zfish_go_association_mixed_path = Path("./network/zfish_proGo_mixed.csv")
    bsub_interactome_path = Path("./network/bsub_proPro.csv")
    bsub_reg_path = Path("./network/bsub_reg.csv")
    bsub_go_association_path = Path("./network/bsub_proGo.csv")
    bsub_go_association_mixed_path = Path("./network/bsub_proGo_mixed.csv")
    yeast_interactome_path = Path("./network/yeast_proPro.csv")
    yeast_reg_path = Path("./network/yeast_reg.csv")
    yeast_go_association_path = Path("./network/yeast_proGo.csv")
    yeast_go_association_mixed_path = Path("./network/yeast_proGo_mixed.csv")
    elegans_interactome_path = Path("./network/elegans_proPro.csv")
    elegans_reg_path = Path("./network/elegans_reg.csv")
    elegans_go_association_path = Path("./network/elegans_proGo.csv")
    elegans_go_association_mixed_path = Path("./network/elegans_proGo_mixed.csv")
    go_depth_path = Path("./network/go_depth.csv")


    output_data_path = Path("./output/data/")
    output_image_path = Path("./output/images/")
    dataset_directory_path = Path("./output/dataset")
    graph_file_path = Path(dataset_directory_path, "graph.pickle")
    go_protein_file_path = Path(dataset_directory_path, "go_protein.pickle")
    protein_file_path = Path(dataset_directory_path, "protein.pickle")

    namespace = ["molecular_function", "biological_process", "cellular_component"]
    sample_size = 10
    repeats = 1
    new_random_lists = True
    print_graphs = True
    no_inferred_edges = True
    go_term_type = [namespace[0],namespace[1],namespace[2]]
    # sample_size: number of samples chosen for positive/negative lists (total is 2xsample_size)
    # repeats: number of times to run all algorithms to obtain an average
    # new_random_lists: if the pos/neg lists already exist (False) or to create new pos/neg lists using sample size and repeats (True)
    # print_graphs: to output data as graphs (True) or not (False)
    # no_inferred_edges: To use inferred edges (False) or to remove inferred edges (True)
    # go_term_type: When new_random_lists is True, change to include the namespaces used in the sample
    
    testing_output_data_path = Path("./output/data/")
    testing_output_image_path = Path("./output/images/")
    testing_input_directory_path = Path("./tests/testing-dataset/")
    testing_graph_file_path = Path(testing_input_directory_path, "graph.pickle")

    
    short_name = ""
    # When using previously created lists, this uses the go_term_types in the file name to find which types are used
    if new_random_lists == False:
        go_term_type = []
        data_dir = sorted(os.listdir(dataset_directory_path))
        for j in data_dir:
            if j.startswith('rep_0_neg'):
                file = j
        file = file.replace(".", "_")
        file = file.split("_")
        if "mol" in file:
            go_term_type.append(namespace[0])
            short_name = short_name + "_mol"
        if "bio" in file:
            go_term_type.append(namespace[1])
            short_name = short_name + "_bio"
        if "cel" in file:
            go_term_type.append(namespace[2])
            short_name = short_name + "_cel"
    else: 
        if namespace[0] in go_term_type:
            short_name = short_name + "_mol"
        if namespace[1] in go_term_type:
            short_name = short_name + "_bio"
        if namespace[2] in go_term_type:
            short_name = short_name + "_cel"

    interactome_columns = [0, 1]
    interactome = read_specific_columns(elegans_interactome_path, interactome_columns, ",")
    regulatory_interactome = read_specific_columns(elegans_reg_path, interactome_columns, ",")
    go_inferred_columns = [0, 2, 3]
    #Adds relationship_type column
    if no_inferred_edges: 
        go_inferred_columns.append(1)
        
    go_protein_pairs = read_pro_go_data(
        elegans_go_association_mixed_path, go_inferred_columns, go_term_type, ","
    )
    #Uses relationship_type column to sort through which proGO edges are inferred 
    if no_inferred_edges:
        temp = []
        for i in go_protein_pairs:
            if i[3] != "inferred_from_descendant":
                temp.append(i)
        go_protein_pairs = temp
    

    depth_columns = [0,1,2]
    go_depth_dict = read_go_depth_data(go_depth_path, depth_columns, go_term_type, ',')

    protein_list = []

    # Generate a standard graph using the pro-pro, regulatory, and pro-go interactions
    G, protein_list = create_mixed_network(interactome,regulatory_interactome, go_protein_pairs, go_depth_dict)
    export_graph_to_pickle(G, graph_file_path)
    # Creates a graph with only protein-protein edges (used for RandomWalkV4)
    # P, protein_list = create_only_protein_network(interactome,regulatory_interactome, go_protein_pairs, go_depth_dict)
    # export_graph_to_pickle(P, "./output/dataset/protein.pickle")
    # Creates a graph with only protein-GO term edges (used for RandomWalkV5)
    # D, protein_list = create_go_protein_only_network(interactome,regulatory_interactome, go_protein_pairs, go_depth_dict)
    # export_graph_to_pickle(D, "./output/dataset/go_protein.pickle")

    # sys.exit()
    
    # Define algorithm classes and their names
    algorithm_classes = {
        "OverlappingNeighbors": OverlappingNeighbors,
        "OverlappingNeighborsV2": OverlappingNeighborsV2,
        "OverlappingNeighborsV3": OverlappingNeighborsV3,
        "ProteinDegree": ProteinDegree,
        "ProteinDegreeV2": ProteinDegreeV2,
        "ProteinDegreeV3": ProteinDegreeV3,
        "SampleAlgorithm": SampleAlgorithm,
        "HypergeometricDistribution": HypergeometricDistribution,
        "HypergeometricDistributionV2": HypergeometricDistributionV2,
        "RandomWalk": RandomWalk, 
        "RandomWalkV2": RandomWalkV2, 
        "RandomWalkV3": RandomWalkV3, 
        # "RandomWalkV4": RandomWalkV4,   #need protein-only network
        # "RandomWalkV5": RandomWalkV5,     #need protein-goterm only network
    }

    run_workflow(
        algorithm_classes,
        go_protein_pairs,
        sample_size,
        protein_list,
        graph_file_path,       #make sure you have the correct file path
        dataset_directory_path,
        output_data_path,
        output_image_path,
        repeats,
        new_random_lists,
        short_name,
        print_graphs,
    )

    sys.exit()


if __name__ == "__main__":
    main()
