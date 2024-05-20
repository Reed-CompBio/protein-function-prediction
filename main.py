import matplotlib.pyplot as plt
import networkx as nx
from random import sample
import pandas as pd
import numpy as np
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
from pathlib import Path
from algorithms.degree_function import degree_function
from algorithms.overlapping_neighbors import overlapping_neighbors
from tools.helper import print_progress
import os


def create_ppi_network(fly_interactome, fly_GO_term):
    print("Initializing network")
    i = 1
    total_progress = len(fly_interactome) + len(fly_GO_term)
    G = nx.Graph()
    protein_protein_edge = 0
    protein_go_edge = 0
    protein_node = 0
    go_node = 0

    # go through fly interactome, add a new node if it doesnt exists already, then add their physical interactions as edges
    for line in fly_interactome:
        if not G.has_node(line[2]):
            G.add_node(line[2], name=line[0], type="protein")
            protein_node += 1

        if not G.has_node(line[3]):
            G.add_node(line[3], name=line[1], type="protein")
            protein_node += 1

        G.add_edge(line[2], line[3], type="protein_protein")
        protein_protein_edge += 1
        print_progress(i, total_progress)
        i += 1

    # Proteins annotated with a GO term have an edge to a GO term node
    for line in fly_GO_term:
        if not G.has_node(line[1]):
            G.add_node(line[1], type="go_term")
            go_node += 1

        G.add_edge(line[1], line[0], type="protein_go_term")
        protein_go_edge += 1
        print_progress(i, total_progress)
        i += 1

    print("")
    print("")
    print("network summary")

    print("protein-protein edge count: ", protein_protein_edge)
    print("protein-go edge count: ", protein_go_edge)
    print("protein node count: ", protein_node)
    print("go node count: ", go_node)
    print("total edge count: ", len(G.edges()))
    print("total node count: ", len(G.nodes()))

    return G


def read_specific_columns(file_path, columns):
    try:
        with open(file_path, "r") as file:
            next(file)
            data = []
            for line in file:
                parts = line.strip().split("\t")
                selected_columns = []
                for col in columns:
                    selected_columns.append(parts[col])
                data.append(selected_columns)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def main():
    print("Starting workflow")
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("output/data"):
        os.makedirs("output/data")
    if not os.path.exists("output/images"):
        os.makedirs("output/images")

    interactome_path = Path("./network/interactome-flybase-collapsed-weighted.txt")
    go_path = Path("./network/gene_association.fb")
    output_data_path = Path("./output/data/overlapping_neighbors_output.csv")
    output_image_path = Path("./output/images/overlapping_neighbors_roc.png")
    sample_size = 5000

    output_data_path = Path("./output/data/degree_output.csv")
    output_image_path = Path("./output/images/degree_roc.png")

    interactome_columns = [0, 1, 4, 5]
    interactome = read_specific_columns(interactome_path, interactome_columns)

    go_columns = [1, 4]
    go_terms = read_specific_columns(go_path, go_columns)

    G = create_ppi_network(interactome, go_terms)

    (
        overlapping_neighbors_data,
        overlapping_neighbors_y_true,
        overlapping_neighbors_y_score,
    ) = overlapping_neighbors(go_terms, output_data_path, sample_size, G)

    degree_function_data, degree_function_y_true, degree_function_y_score = (
        degree_function(go_terms, output_data_path, sample_size, G)
    )

    # Compute ROC curve and ROC area for the first classifier
    fpr1, tpr1, thresholds1 = roc_curve(
        overlapping_neighbors_y_true, overlapping_neighbors_y_score
    )
    roc_auc1 = auc(fpr1, tpr1)

    # Compute ROC curve and ROC area for the second classifier
    fpr2, tpr2, thresholds2 = roc_curve(degree_function_y_true, degree_function_y_score)
    roc_auc2 = auc(fpr2, tpr2)

    # Compute precision-recall curve and area under the curve for the first classifier
    precision1, recall1, thresholds1 = precision_recall_curve(
        overlapping_neighbors_y_true, overlapping_neighbors_y_score
    )
    pr_auc1 = auc(recall1, precision1)

    # Compute precision-recall curve and area under the curve for the second classifier
    precision2, recall2, thresholds2 = precision_recall_curve(
        degree_function_y_true, degree_function_y_score
    )
    pr_auc2 = auc(recall2, precision2)

    # Plot ROC Curve for both classifiers
    plt.figure()
    plt.plot(
        fpr1,
        tpr1,
        color="darkorange",
        lw=2,
        label="Overlapping Neighbors (area = %0.2f)" % roc_auc1,
    )
    plt.plot(
        fpr2,
        tpr2,
        color="blue",
        lw=2,
        label="Degree Function (area = %0.2f)" % roc_auc2,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig("./output/images/multiple_roc_curves.png")
    plt.show()

    plt.figure()
    plt.plot(
        recall1,
        precision1,
        color="darkorange",
        lw=2,
        label="Overlapping Neighbor (area = %0.2f)" % pr_auc1,
    )
    plt.plot(
        recall2,
        precision2,
        color="blue",
        lw=2,
        label="Degree Function (area = %0.2f)" % pr_auc2,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig("./output/images/multiple_pr_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
