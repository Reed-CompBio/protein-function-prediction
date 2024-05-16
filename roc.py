import matplotlib.pyplot as plt
import networkx as nx
from random import sample
import pandas as pd
import numpy as np
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import random
import time
from sklearn.metrics import roc_curve, auc





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

    # Proteins annotated with a GO term have an edge to a GO term node
    for line in fly_GO_term:
        if not G.has_node(line[1]):
            G.add_node(line[1], type="go_term")
            go_node += 1

        G.add_edge(line[1], line[0], type="protein_go_term")
        protein_go_edge += 1

    print("protein-protein edge count: ", protein_protein_edge)
    print("protein-go edge count: ", protein_go_edge)
    print("protein node count: ", protein_node)
    print("go node count: ", go_node)

    return G

def getNeighbors(G: nx.Graph, node, edgeType):
    res = G.edges(node, data=True)
    neighbors = []
    for edge in res:
        if(edge[2]["type"] == edgeType):
            neighborNode = [edge[1], edge[2]]
            neighbors.append(neighborNode)

    return neighbors

def getGoAnnotatedProteinCount(G: nx.Graph, nodeList, goTerm):
    count = 0
    for element in nodeList:
        if(G.has_edge(element[0], goTerm)):
            count+=1
    return count

def print_progress(current, total, bar_length=40):
    # Calculate the progress as a percentage
    percent = float(current) / total
    # Determine the number of hash marks in the progress bar
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    # Construct the progress bar string
    progress_bar = f"[{arrow + spaces}] {int(round(percent * 100))}%"

    # Print the progress bar, overwriting the previous line
    print(f'\r{progress_bar}', end='')

def main():
    colorama_init()

    flybase_interactome_file_path = "./interactome-flybase-collapsed-weighted.txt"
    gene_association_file_path = "./gene_association.fb"

    print("-" * 65)
    print("network summary")

    flybase_columns = [0, 1, 4, 5]
    fly_interactome = read_specific_columns(
        flybase_interactome_file_path, flybase_columns
    )

    fly_GO_columns = [1, 4]
    fly_GO_term = read_specific_columns(gene_association_file_path, fly_GO_columns)

    G = create_ppi_network(fly_interactome, fly_GO_term)

    print("total edge count: ", len(G.edges()))
    print("total node count: ", len(G.nodes()))

    positiveProteinGoTermPairs = []
    negativeProteinGoTermPairs = []
    d = {
        "protein": [],
        "goTerm": [],
        "proteinNeighbor": [],
        "goProteinEdge": [],
        "goEdge": [],
        "fScore": [],
    }

    print("-" * 65)
    print("Sampling Data")

    totalSamples = 5000

    for edge in sample(list(fly_GO_term), totalSamples):
        positiveProteinGoTermPairs.append(edge)

    tempPairs = positiveProteinGoTermPairs.copy()
    i = 0
    for edge in positiveProteinGoTermPairs:
        sampleEdge = random.choice(tempPairs)
        tempPairs.remove(sampleEdge)
        #removes duplicate proteins and if a protein has a corresponding edge to the GO term in the network
        while (sampleEdge[0] == edge[0] and not G.has_edge(sampleEdge[0], edge[1])):
            print("Found a duplicate or has an exisitng edge")
            tempPairs.append(sampleEdge)
            sampleEdge = random.choice(tempPairs)
            tempPairs.remove(sampleEdge)
        negativeProteinGoTermPairs.append([sampleEdge[0], edge[1]])
        print_progress(i, totalSamples)
        print(i)
        i+=1

    
    print("-" * 65)
    print("Calculating Protein Prediction")

    # have two sets of positive and negative protein-go_term pairs
    # for each pair, calculate the score of how well they predict whether a protein should be annotated to a GO term.
    # 50% of the data are proteins that are annotated to a GO term
    # 50% of the data are proteins that are not annotated to a GO term

    #true postiive rate = sensitivity = (true positives) / (true positives + false negatives)
    #false positive rate = (1 - specificity) = false positives / (false positives + true negatives)

    # precision = goProteinEdgeCount / proteinInterestNeighborCount
    # recall = goProteinEdgeCount / goEdgeCount
    # return 2 * ((precision * recall) / (precision + recall))

    totalScores = {"protein": [], "goTerm": [], "proProNeighbor": [], "goNeighbor": [], "goAnnotatedProProNeighbors": [], "score": [], "label": []}
    i = 0
    for positiveEdge, negativeEdge in zip(positiveProteinGoTermPairs, negativeProteinGoTermPairs):

        positiveProProNeighbor = getNeighbors(G, positiveEdge[0], "protein_protein")
        positiveGoNeighbor = getNeighbors(G, positiveEdge[1], "protein_go_term")
        positiveGoAnnotatedProteinCount =  getGoAnnotatedProteinCount(G, positiveProProNeighbor, positiveEdge[1])
        positiveScore = (positiveGoAnnotatedProteinCount) / (len(positiveProProNeighbor) + len(positiveGoNeighbor))

        negativeProProNeighbor = getNeighbors(G, negativeEdge[0], "protein_protein")
        negativeGoNeighbor = getNeighbors(G, negativeEdge[1], "protein_go_term")
        negativeGoAnnotatedProteinCount = getGoAnnotatedProteinCount(G, negativeProProNeighbor, negativeEdge[1])
        negativeScore = (negativeGoAnnotatedProteinCount) / (len(negativeProProNeighbor) + len(negativeGoNeighbor))


        totalScores["protein"].append(positiveEdge[0])
        totalScores["goTerm"].append(positiveEdge[1])
        totalScores["proProNeighbor"].append(len(positiveProProNeighbor))
        totalScores["goNeighbor"].append(len(positiveGoNeighbor))
        totalScores["goAnnotatedProProNeighbors"].append(positiveGoAnnotatedProteinCount)
        totalScores["score"].append(positiveScore)
        if positiveScore > 0.1:
            totalScores["label"].append("TP")
        else:totalScores["label"].append("FP")


        totalScores["protein"].append(negativeEdge[0])
        totalScores["goTerm"].append(negativeEdge[1])
        totalScores["proProNeighbor"].append(len(negativeProProNeighbor))
        totalScores["goNeighbor"].append(len(negativeGoNeighbor))
        totalScores["goAnnotatedProProNeighbors"].append(negativeGoAnnotatedProteinCount)
        totalScores["score"].append(negativeScore)
        if negativeScore > 0.01:
            totalScores["label"].append("TN")
        else:totalScores["label"].append("FN")

        print_progress(i, totalSamples)
        i+=1

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for label in totalScores["label"]:
        match label:
            case "TP":
                tp+=1
            case "FP":
                fp+=1
            case "TN":
                tn+=1
            case "FN":
                fn+=1
    
    print("")
    print("-" * 65)
    print("Results")

    print("True Positives: ", tp)
    print("False Positives: ", fp)
    print("True Negatives: ", tn)
    print("False Negatives: ", fn)

    y_true = []
    y_scores = []  

    i = 1
    for score in totalScores["score"]:
        if i%2 == 1:
            y_true.append(1)
        else:   
            y_true.append(0)
        y_scores.append(score)
        i+=1 

    # for t,s, in zip(y_true, y_scores):
    #     print(t, s)


    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    totalScoresDf = pd.DataFrame(totalScores)
        
    totalScoresDf.to_csv("totalScoresDf.csv", index=False, sep="\t")


if __name__ == "__main__":
    main()
