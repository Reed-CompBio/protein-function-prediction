from classes.base_algorithm_class import BaseAlgorithm
import networkx as nx
import pandas as pd
import numpy as np
from colorama import init as colorama_init
from colorama import Fore, Back, Style
from pathlib import Path
from tools.helper import print_progress, normalize, get_neighbors


class ProteinDegreeV3(BaseAlgorithm):
    def __init__(self):
        self.y_score = []
        self.y_true = []

    def predict(
        self,
        positive_data_set,
        negative_data_set,
        G: nx.graph,
        output_path,
    ):
        colorama_init()
        print("-" * 65)
        print(Fore.GREEN + Back.BLACK + "degree function predictor algorithm")
        print(Style.RESET_ALL + "")
        data = {
            "protein": [],
            "go_term": [],
            "degree": [],
            "norm_score": [],
            "true_label": [],
        }

        print("")
        print("")
        print("Calculating Protein Prediction")
        i = 1
        for positive_protein, positive_go, negative_protein, negative_go in zip(
            positive_data_set["protein"],
            positive_data_set["go"],
            negative_data_set["protein"],
            negative_data_set["go"],
        ):

            data["protein"].append(positive_protein)
            data["go_term"].append(positive_go)
            data["degree"].append(len(get_neighbors(G, positive_protein, "protein_go_term")))
            data["true_label"].append(1)

            data["protein"].append(negative_protein)
            data["go_term"].append(negative_go)
            data["degree"].append(len(get_neighbors(G, negative_protein, "protein_go_term")))
            data["true_label"].append(0)
            print_progress(i, len(positive_data_set["protein"]))
            i += 1

        normalized_data = normalize(data["degree"])
        for item in normalized_data:
            data["norm_score"].append(item)

        df = pd.DataFrame(data)
        df = df.sort_values(by="norm_score", ascending=False)

        df.to_csv(
            Path(output_path, "protein_degree_data.csv"),
            index=False,
            sep="\t",
        )

        self.y_score = df["norm_score"].to_list()
        self.y_true = df["true_label"].to_list()


def normalize(data):
    data = np.array(data)
    min_val = data.min()
    max_val = data.max()

    if min_val == max_val:
        return np.zeros_like(data)

    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data.tolist()
