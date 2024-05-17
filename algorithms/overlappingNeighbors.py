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


def get_neighbors(G: nx.Graph, node, edgeType):
    res = G.edges(node, data=True)
    neighbors = []
    for edge in res:
        if edge[2]["type"] == edgeType:
            neighborNode = [edge[1], edge[2]]
            neighbors.append(neighborNode)

    return neighbors


def get_go_annotated_protein_count(G: nx.Graph, nodeList, goTerm):
    count = 0
    for element in nodeList:
        if G.has_edge(element[0], goTerm):
            count += 1
    return count


def overlapping_neighbors(
    go_terms: Path, output_data_path: Path, sample_size: int, G: nx.Graph
):
    """
    evaluate overlapping neighbors method on a protein protein interaction network with go term annotation.
    """
    colorama_init()
    print("-" * 65)
    print(Fore.GREEN + Back.BLACK + "overlapping neighbors algorithm")
    print(Style.RESET_ALL + "")

    positive_protein_go_term_pairs = []
    negative_protein_go_term_pairs = []

    print("")
    print("Sampling Data")

    total_samples = sample_size

    for edge in sample(list(go_terms), total_samples):
        positive_protein_go_term_pairs.append(edge)

    temp_pairs = positive_protein_go_term_pairs.copy()
    i = 1
    for edge in positive_protein_go_term_pairs:
        sample_edge = random.choice(temp_pairs)
        temp_pairs.remove(sample_edge)
        # removes duplicate proteins and if a protein has a corresponding edge to the GO term in the network
        while sample_edge[0] == edge[0] and not G.has_edge(sample_edge[0], edge[1]):
            print("Found a duplicate or has an exisitng edge")
            temp_pairs.append(sample_edge)
            sample_edge = random.choice(temp_pairs)
            temp_pairs.remove(sample_edge)
        negative_protein_go_term_pairs.append([sample_edge[0], edge[1]])
        print_progress(i, total_samples)
        i += 1

    print("")
    print("")
    print("Calculating Protein Prediction")

    # have two sets of positive and negative protein-go_term pairs
    # for each pair, calculate the score of how well they predict whether a protein should be annotated to a GO term.
    # 50% of the data are proteins that are annotated to a GO term
    # 50% of the data are proteins that are not annotated to a GO term
    # score equation (1 + number of GoAnnotatedProteinCount) / (number of positiveProProNeighbor + number of positiveGoNeighbor)

    data = {
        "protein": [],
        "go_term": [],
        "pro_pro_neighbor": [],
        "go_neighbor": [],
        "go_annotated_pro_pro_neighbors": [],
        "score": [],
    }
    i = 1
    for positive_edge, negative_edge in zip(
        positive_protein_go_term_pairs, negative_protein_go_term_pairs
    ):

        # calculate the score for the positive set
        positive_pro_pro_neighbor = get_neighbors(
            G, positive_edge[0], "protein_protein"
        )
        positive_go_neighbor = get_neighbors(G, positive_edge[1], "protein_go_term")
        positive_go_annotated_protein_count = get_go_annotated_protein_count(
            G, positive_pro_pro_neighbor, positive_edge[1]
        )
        positive_score = (1 + positive_go_annotated_protein_count) / (
            len(positive_pro_pro_neighbor) + len(positive_go_neighbor)
        )

        # calculate the score for the negative set
        negative_pro_pro_neighbor = get_neighbors(
            G, negative_edge[0], "protein_protein"
        )
        negative_go_neighbor = get_neighbors(G, negative_edge[1], "protein_go_term")
        negative_go_annotated_protein_count = get_go_annotated_protein_count(
            G, negative_pro_pro_neighbor, negative_edge[1]
        )
        negative_score = (1 + negative_go_annotated_protein_count) / (
            len(negative_pro_pro_neighbor) + len(negative_go_neighbor)
        )

        # input positive and negative score to data
        data["protein"].append(positive_edge[0])
        data["go_term"].append(positive_edge[1])
        data["pro_pro_neighbor"].append(len(positive_pro_pro_neighbor))
        data["go_neighbor"].append(len(positive_go_neighbor))
        data["go_annotated_pro_pro_neighbors"].append(
            positive_go_annotated_protein_count
        )
        data["score"].append(positive_score)

        data["protein"].append(negative_edge[0])
        data["go_term"].append(negative_edge[1])
        data["pro_pro_neighbor"].append(len(negative_pro_pro_neighbor))
        data["go_neighbor"].append(len(negative_go_neighbor))
        data["go_annotated_pro_pro_neighbors"].append(
            negative_go_annotated_protein_count
        )
        data["score"].append(negative_score)

        print_progress(i, total_samples)
        i += 1

    # prepare for roc curve by annotating the score by postiive or negative
    y_true = []
    y_scores = []

    i = 1
    for score in data["score"]:
        if i % 2 == 1:
            y_true.append(1)
        else:
            y_true.append(0)
        y_scores.append(score)
        i += 1

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

    df = pd.DataFrame(data)
    df.to_csv(output_data_path, index=False, sep="\t")

    return (data, y_true, y_scores)
