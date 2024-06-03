from classes.base_algorithm_class import BaseAlgorithm
import networkx as nx
import pandas as pd
import numpy as np
from colorama import init as colorama_init
from colorama import Fore, Back, Style
from pathlib import Path
from tools.helper import print_progress, normalize
from tools.workflow import get_datasets


class ProteinDegree(BaseAlgorithm):
    def __init__(self):
        self.y_score = []
        self.y_true = []

    def predict(
        self,
        input_directory_path,
        G: nx.graph,
        output_path,
    ):
        colorama_init()
        data = {
            "protein": [],
            "go_term": [],
            "degree": [],
            "norm_score": [],
            "true_label": [],
        }

        positive_dataset, negative_dataset = get_datasets(input_directory_path)

        i = 1
        for positive_protein, positive_go, negative_protein, negative_go in zip(
            positive_dataset["protein"],
            positive_dataset["go"],
            negative_dataset["protein"],
            negative_dataset["go"],
        ):

            data["protein"].append(positive_protein)
            data["go_term"].append(positive_go)
            data["degree"].append(G.degree(positive_protein))
            data["true_label"].append(1)

            data["protein"].append(negative_protein)
            data["go_term"].append(negative_go)
            data["degree"].append(G.degree(negative_protein))
            data["true_label"].append(0)
            print_progress(i, len(positive_dataset["protein"]))
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
