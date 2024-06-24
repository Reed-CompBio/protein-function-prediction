from classes.base_algorithm_class import BaseAlgorithm
import networkx as nx
import pandas as pd
from colorama import init as colorama_init
from colorama import Fore, Back, Style
from pathlib import Path
from tools.helper import print_progress, normalize, import_graph_from_pickle
import random
from tools.workflow import get_datasets


class SampleAlgorithm(BaseAlgorithm):
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
        output_data_directory,
        rep_num,
        name,
    ):
        """
        evaluate a random approach method on a protein protein interaction network with go term annotation.
        """
        # initialize an objext that will hold the prediction data
        data = {
            "protein": [],
            "go_term": [],
            "score": [],
            "norm_score": [],
            "true_label": [],
        }

        positive_dataset, negative_dataset = get_datasets(input_directory_path, rep_num, name)
        G = import_graph_from_pickle(graph_file_path)

        i = 1
        # iterate through the positive and negative dataset and calculate the method's prediction score
        for positive_protein, positive_go, negative_protein, negative_go in zip(
            positive_dataset["protein"],
            positive_dataset["go"],
            negative_dataset["protein"],
            negative_dataset["go"],
        ):
            # prediction logic for the positive and negative data set entry
            positive_score = random.random()
            negative_score = random.random()

            # input the positive data
            data["protein"].append(positive_protein)
            data["go_term"].append(positive_go)
            data["score"].append(positive_score)
            data["true_label"].append(1)

            # input the negative data
            data["protein"].append(negative_protein)
            data["go_term"].append(negative_go)
            data["score"].append(negative_score)
            data["true_label"].append(0)
            print_progress(i, len(positive_dataset["protein"]))
            i += 1

        # need to normalise the data
        normalized_data = normalize(data["score"])
        for item in normalized_data:
            data["norm_score"].append(item)

        # convert the data to a pandas dataframe and sort by highest norm_score to lowest
        df = pd.DataFrame(data)
        df = df.sort_values(by="norm_score", ascending=False)

        # output the result data
        df.to_csv(
            Path(output_data_directory, "sample_algorithm_data.csv"),
            index=False,
            sep="\t",
        )

        # ALWAYS set the class attribute variables to the norm_score and true_label
        y_score = df["norm_score"].to_list()
        y_true = df["true_label"].to_list()

        return y_score, y_true
