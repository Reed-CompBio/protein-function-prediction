from classes.base_algorithm_class import BaseAlgorithm
import networkx as nx
import pandas as pd
from colorama import init as colorama_init
from colorama import Fore, Back, Style
from pathlib import Path
from tools.helper import print_progress, normalize, import_graph_from_pickle
from tools.workflow import get_datasets


class OverlappingNeighbors(BaseAlgorithm):
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
        """
        evaluate overlapping neighbors method on a protein protein interaction network with go term annotation.
        """
        colorama_init()

        # have two sets of positive and negative protein-go_term pairs
        # for each pair, calculate the score of how well they predict whether a protein should be annotated to a GO term.
        # 50% of the data are proteins that are annotated to a GO term
        # 50% of the data are proteins that are not annotated to a GO term
        # score equation (1 + number of ProProNeighbor that are annotated to the go term) / (number of ProProNeighbor + number of GoNeighbor)

        data = {
            "protein": [],
            "go_term": [],
            "protein_neighbor": [],
            "go_neighbor": [],
            "go_annotated_protein_neighbors": [],
            "score": [],
            "norm_score": [],
            "true_label": [],
        }

        positive_dataset, negative_dataset = get_datasets(
            input_directory_path, rep_num, name
        )
        G = import_graph_from_pickle(graph_file_path)
        i = 1
        for positive_protein, positive_go in zip(
            positive_dataset["protein"],
            positive_dataset["go"],
        ):
            # calculate the score for the positive set
            positive_protein_neighbor = get_neighbors(
                G, positive_protein, ["protein_protein", "regulatory"]
            )

            # print("\nPositive protein neighbors: " + str(positive_pro_pro_neighbor))
            positive_go_neighbor = get_neighbors(G, positive_go, ["protein_go_term"])
            positive_go_annotated_protein_neighbor_count = (
                get_go_annotated_protein_neighbor_count(
                    G, positive_protein_neighbor, positive_go
                )
            )

            if len(positive_protein_neighbor) == 0:
                positive_score = 0
            else:
                positive_score = (1 + positive_go_annotated_protein_neighbor_count) / (
                    len(positive_protein_neighbor) + len(positive_go_neighbor)
                )

            # input positive and negative score to data
            data["protein"].append(positive_protein)
            data["go_term"].append(positive_go)
            data["protein_neighbor"].append(len(positive_protein_neighbor))
            data["go_neighbor"].append(len(positive_go_neighbor))
            data["go_annotated_protein_neighbors"].append(
                positive_go_annotated_protein_neighbor_count
            )
            data["score"].append(positive_score)
            data["true_label"].append(1)

            print_progress(i, len(positive_dataset["protein"]))
            i += 1

        for negative_protein, negative_go in zip(
            negative_dataset["protein"],
            negative_dataset["go"],
        ):

            # calculate the score for the negative set
            negative_protein_neighbor = get_neighbors(
                G, negative_protein, "protein_protein"
            )
            negative_go_neighbor = get_neighbors(G, negative_go, "protein_go_term")
            negative_go_annotated_protein_neighbor_count = (
                get_go_annotated_protein_neighbor_count(
                    G, negative_protein_neighbor, negative_go
                )
            )

            if len(negative_protein_neighbor) == 0:
                negative_score = 0
            else:
                negative_score = (1 + negative_go_annotated_protein_neighbor_count) / (
                    len(negative_protein_neighbor) + len(negative_go_neighbor)
                )

            data["protein"].append(negative_protein)
            data["go_term"].append(negative_go)
            data["protein_neighbor"].append(len(negative_protein_neighbor))
            data["go_neighbor"].append(len(negative_go_neighbor))
            data["go_annotated_protein_neighbors"].append(
                negative_go_annotated_protein_neighbor_count
            )
            data["score"].append(negative_score)
            data["true_label"].append(0)

            print_progress(i, len(negative_dataset["protein"]))
            i += 1

        normalized_data = normalize(data["score"])
        for item in normalized_data:
            data["norm_score"].append(item)

        df = pd.DataFrame(data)
        df = df.sort_values(by="norm_score", ascending=False)

        df.to_csv(
            Path(output_path, "overlapping_neighbor_data.csv"),
            index=False,
            sep="\t",
        )

        y_score = df["norm_score"].to_list()
        y_true = df["true_label"].to_list()

        return y_score, y_true


def get_neighbors(G: nx.DiGraph, node, edgeTypes):
    res = G.edges(node, data=True)
    neighbors = []
    for edge in res:
        if edge[2]["type"] in edgeTypes:
            neighborNode = [edge[1], edge[2]]
            neighbors.append(neighborNode)

    return neighbors


def get_go_annotated_protein_neighbor_count(G: nx.DiGraph, nodeList, goTerm):
    count = 0
    for element in nodeList:
        if G.has_edge(element[0], goTerm):
            count += 1
    return count
