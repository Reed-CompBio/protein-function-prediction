from classes.overlapping_neighbors_class import OverlappingNeighbors
from classes.protein_degree_class import ProteinDegree
import random
import matplotlib.pyplot as plt
import networkx as nx
from random import sample
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from pathlib import Path
from algorithms.degree_function import degree_function
from algorithms.overlapping_neighbors import overlapping_neighbors
from tools.helper import print_progress
import os
import sys
import networkx as nx
import pandas as pd
import numpy as np
from colorama import init as colorama_init
from colorama import Fore, Back, Style
from sklearn.metrics import roc_curve, auc, f1_score
from pathlib import Path
from tools.helper import print_progress


def create_ppi_network(fly_interactome, fly_GO_term):
    print("Initializing network")
    i = 1
    total_progress = len(fly_interactome) + len(fly_GO_term)
    G = nx.Graph()
    protein_protein_edge = 0
    protein_go_edge = 0
    protein_node = 0
    go_node = 0
    protein_list = []
    go_term_list = []

    # go through fly interactome, add a new node if it doesnt exists already, then add their physical interactions as edges
    for line in fly_interactome:
        if not G.has_node(line[2]):
            G.add_node(line[2], name=line[0], type="protein")
            protein_list.append({"id": line[2], "name": line[0]})
            protein_node += 1

        if not G.has_node(line[3]):
            G.add_node(line[3], name=line[1], type="protein")
            protein_list.append({"id": line[3], "name": line[1]})
            protein_node += 1

        G.add_edge(line[2], line[3], type="protein_protein")
        protein_protein_edge += 1
        print_progress(i, total_progress)
        i += 1

    # Proteins annotated with a GO term have an edge to a GO term node
    for line in fly_GO_term:
        if not G.has_node(line[1]):
            G.add_node(line[1], type="go_term")
            go_term_list.append(line[1])
            go_node += 1

        if not G.has_node(line[0]):
            G.add_node(line[0], name=line[0], type="protein")
            protein_list.append({"id": line[0], "name": line[0]})
            protein_node += 1

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

    return G, protein_list, go_term_list


def read_specific_columns(file_path, columns, delimit):
    try:
        with open(file_path, "r") as file:
            next(file)
            data = []
            for line in file:
                parts = line.strip().split(delimit)
                selected_columns = []
                for col in columns:
                    selected_columns.append(parts[col].replace('"', ""))
                data.append(selected_columns)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    print("Starting working")

    print("Starting workflow")
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("output/data"):
        os.makedirs("output/data")
    if not os.path.exists("output/images"):
        os.makedirs("output/images")
    if not os.path.exists("input"):
        os.makedirs("input")

    interactome_path = Path("./network/interactome-flybase-collapsed-weighted.txt")
    go_association_path = Path("./network/fly_proGo.csv")
    output_data_path = Path("./output/data/")
    output_image_path = Path("./output/images/")
    sample_size = 100

    interactome_columns = [0, 1, 4, 5]
    interactome = read_specific_columns(interactome_path, interactome_columns, "\t")

    go_inferred_columns = [0, 2]
    go_protein_pairs = read_specific_columns(
        go_association_path, go_inferred_columns, ","
    )

    protein_list = []
    go_term_list = []

    G, protein_list, go_term_list = create_ppi_network(interactome, go_protein_pairs)

    positive_dataset = {"protein": [], "go": []}
    negative_dataset = {"protein": [], "go": []}

    print("")
    print("Sampling Data")

    # sample the data
    for edge in sample(list(go_protein_pairs), sample_size):
        positive_dataset["protein"].append(edge[0])
        positive_dataset["go"].append(edge[1])

    i = 1

    for protein, go in zip(
        positive_dataset["protein"], positive_dataset["go"]
    ):
        sample_edge = random.choice(protein_list)
        # print(sample_edge, edge)
        # removes duplicate proteins and if a protein has a corresponding edge to the GO term in the network
        while G.has_edge(sample_edge["id"], go):
            # print("has an exisitng edge")
            sample_edge = random.choice(protein_list)
        negative_dataset["protein"].append(sample_edge["id"])
        negative_dataset["go"].append(go)
        print_progress(i, sample_size)
        i += 1

    positive_df = pd.DataFrame(positive_dataset)
    negative_df = pd.DataFrame(negative_dataset)

    positive_df.to_csv(
        "./input/positive_protein_go_term_pairs.csv", index=False, sep="\t"
    )
    negative_df.to_csv(
        "./input/negative_protein_go_term_pairs.csv", index=False, sep="\t"
    )

    overlapping_neighbors = OverlappingNeighbors()

    overlapping_neighbors.predict(positive_dataset, negative_dataset, sample_size, G, output_data_path)

    overlapping_neighbors_y_score = overlapping_neighbors.y_scores
    overlapping_neighbors_y_true = overlapping_neighbors.y_true

    protein_degree = ProteinDegree()

    protein_degree.predict(positive_dataset, negative_dataset, sample_size, G, output_data_path)

    protein_degree_y_score = protein_degree.y_scores
    protein_degree_y_true = protein_degree.y_true

    fpr1, tpr1, thresholds_1 = roc_curve(overlapping_neighbors_y_true, overlapping_neighbors_y_score)
    roc_auc1 = auc(fpr1, tpr1)

    fpr2, tpr2, thresholds_2 = roc_curve(protein_degree_y_true, protein_degree_y_score)
    roc_auc2 = auc(fpr2, tpr2)

    print("")
    print("")
    print("Calculating optimal thresholds")

    # 1. Maximize the Youden’s J Statistic
    youden_j_1 = tpr1 - fpr1
    optimal_index_youden_1 = np.argmax(youden_j_1)
    optimal_threshold_youden_1 = thresholds_1[optimal_index_youden_1]

    i = 1
    # 2. Maximize the F1 Score
    # For each threshold, compute the F1 score
    f1_scores = []
    for threshold in thresholds_1:
        y_pred = (overlapping_neighbors_y_score >= threshold).astype(int)
        f1 = f1_score(overlapping_neighbors_y_true, y_pred)
        f1_scores.append(f1)
        print_progress(i, len(thresholds_1))
        i += 1
    optimal_index_f1 = np.argmax(f1_scores)
    optimal_threshold_f1_1 = thresholds_1[optimal_index_f1]

    # 3. Minimize the Distance to (0, 1) on the ROC Curve
    distances = np.sqrt((1 - tpr1) ** 2 + fpr1**2)
    optimal_index_distance = np.argmin(distances)
    optimal_threshold_distance_1 = thresholds_1[optimal_index_distance]

    print("")
    print("")
    print("-" * 65)
    print("Results")
    print("")

    # Print the optimal thresholds for each approach
    print(Fore.YELLOW + "Optimal Threshold (Youden's J):", optimal_threshold_youden_1)
    print("Optimal Threshold (F1 Score):", optimal_threshold_f1_1)
    print("Optimal Threshold (Min Distance to (0,1)):", optimal_threshold_distance_1)
    print(Style.RESET_ALL + "")

    # 1. Maximize the Youden’s J Statistic
    youden_j_2 = tpr2 - fpr2
    optimal_index_youden_2 = np.argmax(youden_j_2)
    optimal_threshold_youden_2 = thresholds_2[optimal_index_youden_2]

    i = 1
    # 2. Maximize the F1 Score
    # For each threshold, compute the F1 score
    f1_scores = []
    for threshold in thresholds_2:
        y_pred = (protein_degree_y_score >= threshold).astype(int)
        f1 = f1_score(protein_degree_y_true, y_pred)
        f1_scores.append(f1)
        print_progress(i, len(thresholds_2))
        i += 1
    optimal_index_f2 = np.argmax(f1_scores)
    optimal_threshold_f2 = thresholds_2[optimal_index_f2]

    # 3. Minimize the Distance to (0, 1) on the ROC Curve
    distances = np.sqrt((1 - tpr2) ** 2 + fpr2**2)
    optimal_index_distance = np.argmin(distances)
    optimal_threshold_distance_2 = thresholds_2[optimal_index_distance]

    print("")
    print("")
    print("-" * 65)
    print("Results")
    print("")

    # Print the optimal thresholds for each approach
    print(Fore.YELLOW + "Optimal Threshold (Youden's J):", optimal_threshold_youden_2)
    print("Optimal Threshold (F1 Score):", optimal_threshold_f2)
    print("Optimal Threshold (Min Distance to (0,1)):", optimal_threshold_distance_2)
    print(Style.RESET_ALL + "")

    # Compute ROC curve and ROC area for the first classifier
    fpr1, tpr1, thresholds1 = roc_curve(
        overlapping_neighbors_y_true, overlapping_neighbors_y_score
    )
    roc_auc1 = auc(fpr1, tpr1)


    # Compute precision-recall curve and area under the curve for the first classifier
    precision1, recall1, thresholds1 = precision_recall_curve(
        overlapping_neighbors_y_true, overlapping_neighbors_y_score
    )
    pr_auc1 = auc(recall1, precision1)

    # Compute ROC curve and ROC area for the second classifier
    fpr2, tpr2, thresholds2 = roc_curve(
        protein_degree_y_true, protein_degree_y_score
    )
    roc_auc2 = auc(fpr2, tpr2)


    # Compute precision-recall curve and area under the curve for the second classifier
    precision2, recall2, thresholds2 = precision_recall_curve(
        protein_degree_y_true, protein_degree_y_score
    )
    pr_auc2 = auc(recall1, precision1)

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
        color="purple",
        lw=2,
        label="Protein Degree (area = %0.2f)" % roc_auc2,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(Path(output_image_path, "multiple_roc_curves.png"))
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
        color="purple",
        lw=2,
        label="Protein Degree (area = %0.2f)" % pr_auc2,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(Path(output_image_path, "multiple_pr_curves.png"))
    plt.show()


if __name__ == "__main__":
    main()
