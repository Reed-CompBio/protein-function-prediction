from classes.base_algorithm_class import BaseAlgorithm
import networkx as nx
import pandas as pd
import numpy as np
from colorama import init as colorama_init
from colorama import Fore, Back, Style
from pathlib import Path
from tools.helper import print_progress, normalize, import_graph_from_pickle
from tools.workflow import get_datasets


class RandomWalk(BaseAlgorithm):
    def __init__(self):
        self.y_score = []
        self.y_true = []

    def get_y_score(self):
        return self.y_score

    def get_y_true(self):
        return self.y_true

    def set_y_score(self, y_score):
        self.y_score = y_score

    def set_y_true(self, y_true):
        self.y_true = y_true

    def predict(
        self,
        input_directory_path,
        graph_file_path,
        output_path,
        rep_num,
        name,
    ):
        '''
        Uses pagerank to calculate the diffusion score for a given positive and negative protein using the GO term as 
        the personalization. Uses the edge between the positive protein and the GO term.
        '''
        colorama_init()
        data = {
            "protein": [],
            "go_term": [],
            "walk": [],
            "norm_score": [],
            "true_label": [],
        }

        positive_dataset, negative_dataset = get_datasets(input_directory_path, rep_num, name)
        G = import_graph_from_pickle(graph_file_path)

        i = 1
        for positive_protein, positive_go in zip(
            positive_dataset["protein"],
            positive_dataset["go"],
        ):
            #A random walk with restart, likely using pagerank
            p = nx.pagerank(G, alpha=0.7, personalization={positive_go:1}) #Try a version where this is generated for every go term simultaneously?
            
            data["protein"].append(positive_protein)
            data["go_term"].append(positive_go)
            data["walk"].append(p[positive_protein])
            data["true_label"].append(1)

            print_progress(i, len(positive_dataset["protein"]))
            i += 1

        i = 1
        for negative_protein, negative_go in zip(
            negative_dataset["protein"],
            negative_dataset["go"],
        ):
            #A random walk with restart, likely using pagerank
            p = nx.pagerank(G, alpha=0.7, personalization={negative_go:1}) #Try a version where this is generated for every go term simultaneously?

            data["protein"].append(negative_protein)
            data["go_term"].append(negative_go)
            data["walk"].append(p[negative_protein])
            data["true_label"].append(0)

            print_progress(i, len(negative_dataset["protein"]))
            i += 1

        normalized_data = normalize(data["walk"])
        for item in normalized_data:
            data["norm_score"].append(item)
        df = pd.DataFrame(data)
        df = df.sort_values(by="norm_score", ascending=False)

        df.to_csv(
            Path(output_path, "random_walk_data.csv"),
            index=False,
            sep="\t",
        )

        y_score = df["norm_score"].to_list()
        y_true = df["true_label"].to_list()

        return y_score, y_true

def normalize(data):
    data = np.array(data)
    min_val = data.min()
    max_val = data.max()

    if min_val == max_val:
        return np.zeros_like(data)

    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data.tolist()
