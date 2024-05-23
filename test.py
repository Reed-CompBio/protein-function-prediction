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
from tools.helper import (
    create_ppi_network,
    read_specific_columns,
    generate_random_colors,
)
from tools.workflow import run_workflow


def main():
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
    sample_size = 10

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

    for protein, go in zip(positive_dataset["protein"], positive_dataset["go"]):
        sample_edge = random.choice(protein_list)
        # removes if a protein has a corresponding edge to the GO term in the network
        while G.has_edge(sample_edge["id"], go):
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

    # Define algorithm classes and their names
    algorithm_classes = {
        "OverlappingNeighbors": OverlappingNeighbors,
        "ProteinDegree": ProteinDegree,
    }

    results = run_workflow(
        algorithm_classes, positive_dataset, negative_dataset, G, output_data_path
    )

    # Calculate thresholding for each method w/ three threshold metrics
    for algorithm_name, metrics in results.items():
        print("")
        print(f"Calculating optimal thresholds: {algorithm_name}")
        # 1. Maximize the Youden’s J Statistic
        youden_j = metrics["tpr"] - metrics["fpr"]
        optimal_index_youden = np.argmax(youden_j)
        optimal_threshold_youden = metrics["thresholds"][optimal_index_youden]

        i = 1
        # 2. Maximize the F1 Score
        # For each threshold, compute the F1 score
        f1_scores = []
        for threshold in metrics["thresholds"]:
            y_pred = (metrics["y_score"] >= threshold).astype(int)
            f1 = f1_score(metrics["y_true"], y_pred)
            f1_scores.append(f1)
            i += 1
        optimal_index_f1 = np.argmax(f1_scores)
        optimal_threshold_f1 = metrics["thresholds"][optimal_index_f1]

        # 3. Minimize the Distance to (0, 1) on the ROC Curve
        distances = np.sqrt((1 - metrics["tpr"]) ** 2 + metrics["fpr"] ** 2)
        optimal_index_distance = np.argmin(distances)
        optimal_threshold_distance = metrics["thresholds"][optimal_index_distance]

        # Print the optimal thresholds for each approach
        print(Fore.YELLOW + "Optimal Threshold (Youden's J):", optimal_threshold_youden)
        print("Optimal Threshold (F1 Score):", optimal_threshold_f1)
        print("Optimal Threshold (Min Distance to (0,1)):", optimal_threshold_distance)
        print(Style.RESET_ALL + "")

    # Generate ROC and PR figures to compare methods

    colors = generate_random_colors(2)
    i = 0
    plt.figure()
    for algorithm_name, metrics in results.items():
        plt.plot(
            metrics["fpr"],
            metrics["tpr"],
            color=colors[i],
            lw=2,
            label=f"{algorithm_name} (area = %0.2f)" % metrics["roc_auc"],
        )
        i += 1

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(Path(output_image_path, "multiple_roc_curves.png"))
    plt.show()

    i = 0
    plt.figure()
    for algorithm_name, metrics in results.items():
        plt.plot(
            metrics["recall"],
            metrics["precision"],
            color=colors[i],
            lw=2,
            label=f"{algorithm_name} (area = %0.2f)" % metrics["pr_auc"],
        )
        i += 1
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(Path(output_image_path, "multiple_pr_curves.png"))
    plt.show()

    sys.exit()


if __name__ == "__main__":
    main()
