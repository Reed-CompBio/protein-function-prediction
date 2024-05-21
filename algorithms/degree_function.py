import matplotlib.pyplot as plt
import networkx as nx
from random import sample
import pandas as pd
import numpy as np
from colorama import init as colorama_init
from colorama import Fore, Back, Style
import random
from sklearn.metrics import roc_curve, auc, f1_score
from pathlib import Path
from tools.helper import print_progress


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


def create_ppi_network(fly_interactome, fly_GO_term):
    print("")
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


def normalize(data):
    data = np.array(data)
    min_val = data.min()
    max_val = data.max()

    if min_val == max_val:
        return np.zeros_like(data)

    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data.tolist()


def degree_function(
    positive_protein_go_term_pairs: list,
    negative_protein_go_term_pairs: list,
    output_data_path: Path,
    sample_size: int,
    G: nx.Graph,
):
    colorama_init()
    print("-" * 65)
    print(Fore.GREEN + Back.BLACK + "degree function predictor algorithm")
    print(Style.RESET_ALL + "")
    data = {"protein": [], "go_term": [], "degree": [], "score": [], "true_label": []}

    print("")
    print("Sampling Data")

    print("")
    print("")
    print("Calculating Protein Prediction")

    for positive_protein, positive_go, negative_protein, negative_go in zip(
        positive_protein_go_term_pairs["protein"],
        positive_protein_go_term_pairs["go"],
        negative_protein_go_term_pairs["protein"],
        negative_protein_go_term_pairs["go"],
    ):

        data["protein"].append(positive_protein)
        data["go_term"].append(positive_go)
        data["degree"].append(G.degree(positive_protein))
        data["true_label"].append(1)

        data["protein"].append(negative_protein)
        data["go_term"].append(negative_go)
        data["degree"].append(G.degree(negative_protein))
        data["true_label"].append(0)

    normalized_data = normalize(data["degree"])
    for item in normalized_data:
        data["score"].append(item)

    df = pd.DataFrame(data)
    df = df.sort_values(by="score", ascending=False)

    y_scores = df["score"].to_list()
    y_true = df["true_label"].to_list()

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print("")
    print("")
    print("Calculating optimal thresholds")

    # 1. Maximize the Youden’s J Statistic
    youden_j = tpr - fpr
    optimal_index_youden = np.argmax(youden_j)
    optimal_threshold_youden = thresholds[optimal_index_youden]

    i = 1
    # 2. Maximize the F1 Score
    # For each threshold, compute the F1 score
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
        print_progress(i, len(thresholds))
        i += 1
    optimal_index_f1 = np.argmax(f1_scores)
    optimal_threshold_f1 = thresholds[optimal_index_f1]

    # 3. Minimize the Distance to (0, 1) on the ROC Curve
    distances = np.sqrt((1 - tpr) ** 2 + fpr**2)
    optimal_index_distance = np.argmin(distances)
    optimal_threshold_distance = thresholds[optimal_index_distance]

    print("")
    print("")
    print("-" * 65)
    print("Results")
    print("")
    # Print the optimal thresholds for each approach
    print(Fore.YELLOW + "Optimal Threshold (Youden's J):", optimal_threshold_youden)
    print("Optimal Threshold (F1 Score):", optimal_threshold_f1)
    print("Optimal Threshold (Min Distance to (0,1)):", optimal_threshold_distance)
    print(Style.RESET_ALL + "")

    df.to_csv(Path(output_data_path, "degree_function_data.csv"), index=False, sep="\t")

    return data, y_true, y_scores
