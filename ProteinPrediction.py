import matplotlib.pyplot as plt
import networkx as nx
from random import sample
import pandas as pd
import numpy as np
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style


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
    G = nx.Graph()

    # go through fly interactome, add a new node if it doesnt exists already, then add their physical interactions as edges
    for line in fly_interactome:
        if not G.has_node(line[2]):
            G.add_node(line[2], name=line[0], type="protein")

        if not G.has_node(line[3]):
            G.add_node(line[3], name=line[1], type="protein")

        G.add_edge(line[2], line[3], type="protein_protein")

    # Proteins annotated with a GO term have an edge to a GO term node
    for line in fly_GO_term:
        if not G.has_node(line[1]):
            G.add_node(line[1], type="go_term")
        G.add_edge(line[1], line[0], type="protein_go_term")

    return G


def getProteinProteinNeighbors(G: nx.Graph, protein):
    res = G.edges(protein, data=True)
    K = nx.Graph()
    K.add_edges_from(res)

    N = nx.Graph()

    for edge in K.edges(data=True):
        if edge[2]["type"] == "protein_protein":
            N.add_edge(edge[0], edge[1])

    return N


def getGoNeighborStats(N: nx.Graph, G: nx.Graph, go_term):
    proteinInterestNeighborCount = 0
    goProteinEdgeCount = 0
    goEdgeCount = len(G.edges(go_term))

    for node in N.nodes(data=True):
        if G.has_edge(node[0], go_term):
            goProteinEdgeCount += 1
        proteinInterestNeighborCount += 1

    print("proteinInterestNeighborCount: ", proteinInterestNeighborCount)
    print("goEdgeCount: ", len(G.edges(go_term)))
    print("goProteinEdgeCount: ", goProteinEdgeCount)

    return proteinInterestNeighborCount, goProteinEdgeCount, goEdgeCount


def getFScore(proteinInterestNeighborCount, goProteinEdgeCount, goEdgeCount):
    # precision is # of correct positive results divided by # of predicted positive result
    # recall is # of correct positive results divided by # of actual positive result

    precision = goProteinEdgeCount / proteinInterestNeighborCount
    recall = goProteinEdgeCount / goEdgeCount
    return 2 * ((precision * recall) / (precision + recall))


def main():
    colorama_init()

    flybase_interactome_file_path = "./interactome-flybase-collapsed-weighted.txt"
    gene_association_file_path = "./gene_association.fb"

    flybase_columns = [0, 1, 4, 5]
    fly_interactome = read_specific_columns(
        flybase_interactome_file_path, flybase_columns
    )

    fly_GO_columns = [1, 4]
    fly_GO_term = read_specific_columns(gene_association_file_path, fly_GO_columns)

    G = create_ppi_network(fly_interactome, fly_GO_term)

    proteinGoTermPairs = []
    d = {
        "protein": [],
        "goTerm": [],
        "proteinNeighbor": [],
        "goProteinEdge": [],
        "goEdge": [],
        "fScore": [],
    }

    for edge in sample(list(fly_GO_term), 100):
        proteinGoTermPairs.append(edge)

    i = 1
    for edge in proteinGoTermPairs:
        print("-" * 65)
        print("experiment: ", i)

        protein = edge[0]
        go_term = edge[1]

        print(protein, go_term)
        temp_G = nx.Graph(G)
        temp_G.remove_edge(protein, go_term)

        N = getProteinProteinNeighbors(G, protein)
        proteinInterestNeighborCount, goProteinEdgeCount, goEdgeCount = (
            getGoNeighborStats(N, G, go_term)
        )

        if (
            proteinInterestNeighborCount == 0
            or goProteinEdgeCount == 0
            or goEdgeCount == 0
        ):
            print()
            print(f"{Fore.RED}cannnot compute f-score{Style.RESET_ALL}")
        else:
            fScore = getFScore(
                proteinInterestNeighborCount, goProteinEdgeCount, goEdgeCount
            )
            d["protein"].append(protein)
            d["goTerm"].append(go_term)
            d["proteinNeighbor"].append(proteinInterestNeighborCount)
            d["goProteinEdge"].append(goProteinEdgeCount)
            d["goEdge"].append(goEdgeCount)
            d["fScore"].append(fScore)
            print()
            print(f"{Fore.GREEN}successfully computed f-score{Style.RESET_ALL}")
            print("f-score: ", fScore)
            i += 1

    df = pd.DataFrame(d)

    df.to_csv("out.csv", index=False, sep="\t")


if __name__ == "__main__":
    main()
