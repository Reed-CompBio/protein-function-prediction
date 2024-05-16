import matplotlib.pyplot as plt
import networkx as nx
from random import sample
import pandas as pd
import numpy as np
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import random
from sklearn.metrics import roc_curve, auc, f1_score
from pathlib import Path
from algorithms.degreeFunction import degreeFunction
from algorithms.overlappingNeighbors import overlappingNeighbors
import os



def main():
    if not os.path.exists("output"):
        os.makedirs("output")
        os.makedirs("output/data")
        os.makedirs("output/images")
        print(f"Directory 'output' created.")
    else:
        print(f"Directory 'output' already exists.")

    interactome_path = Path("./network/interactome-flybase-collapsed-weighted.txt")
    go_path = Path("./network/gene_association.fb")
    output_data_path = Path("./output/data/overlapping_neighbors_output.csv")
    output_image_path = Path("./output/images/overlapping_neighbors_roc.png")
    sample_size = 5000

    overlapping_neighbors_data, overlapping_neighbors_y_true, overlapping_neighbors_y_score = (
        overlappingNeighbors(
            interactome_path, go_path, output_data_path, output_image_path, sample_size
        )
    )

    output_data_path = Path("./output/data/degree_output.csv")
    output_image_path = Path("./output/images/degree_roc.png")

    degreeFunctionData, degreeFunctionYTrue, degreeFunctionYScore = degreeFunction(
        interactome_path, go_path, output_data_path, output_image_path, sample_size
    )


    # Compute ROC curve and ROC area for the first classifier
    fpr1, tpr1, thresholds1 = roc_curve(overlapping_neighbors_y_true, overlapping_neighbors_y_score)
    roc_auc1 = auc(fpr1, tpr1)

    # Compute ROC curve and ROC area for the second classifier
    fpr2, tpr2, thresholds2 = roc_curve(degreeFunctionYTrue, degreeFunctionYScore)
    roc_auc2 = auc(fpr2, tpr2)

    # Plot ROC Curve for both classifiers
    plt.figure()
    plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='Overlapping Neighbors (area = %0.2f)' % roc_auc1)
    plt.plot(fpr2, tpr2, color='blue', lw=2, label='Degree Function (area = %0.2f)' % roc_auc2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the plot as an image file
    plt.savefig('./output/images/multiple_roc_curves.png')
    plt.show()


if __name__ == "__main__":
    main()
