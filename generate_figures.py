from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def read_file(filepath: str) -> dict:
    try:
        with open(filepath, "r") as file:
            next(file)
            data = {"y_score": [], "y_true": []}
            for line in file:
                parts = line.strip().split("\t")
                score = float(parts[len(parts) - 2])
                label = int(parts[len(parts) - 1])

                data["y_score"].append(score)
                data["y_true"].append(label)
            return data
    except FileNotFoundError:
        print(f"file {filepath} not found error")
        return None


def get_roc_data(data_df: dict) -> list:
    y = np.array(data_df["y_true"])
    scores = np.array(data_df["y_score"])
    fpr, tpr, threshold = roc_curve(y, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, threshold, roc_auc


def create_plot(ax, x_data: list, y_data: list, auc: float, type: str, color) -> None:
    ax.plot(
        x_data,
        y_data,
        color=color,
        lw=2,
        label=f"{type}(area = %0.2f)" % auc,
    )


def main():
    print("Generating figures")
    species_list = ["elegans", "fly", "bsub", "yeast", "zfish"]
    final_data = defaultdict(list)

    for species in species_list:
        overlapping_path = Path(
            f"./results/final-inferred-complete/{species}/overlapping_neighbor_data.csv"
        )
        hypergeometric_path = Path(
            f"./results/final-inferred-complete/{species}/hypergeometric_distribution.csv"
        )
        degree_path = Path(
            f"./results/final-inferred-complete/{species}/protein_degree_v3_data.csv"
        )
        rw_path = Path(
            f"./results/final-inferred-complete/{species}/random_walk_data_v2.csv"
        )

        species_path = [overlapping_path, hypergeometric_path, degree_path, rw_path]

        methods = []
        for path in species_path:
            data = read_file(path)
            methods.append(data)

        # calculate AUC values
        fpr_list = []
        tpr_list = []
        threshold_list = []
        roc_auc_list = []

        for data in methods:
            fpr, tpr, threshold, roc_auc = get_roc_data(data)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            threshold_list.append(threshold)
            roc_auc_list.append(roc_auc)

        species_data = {
            "fpr": fpr_list,
            "tpr": tpr_list,
            "roc": roc_auc_list,
            "method": ["Overlapping", "Hypergeometric", "Degree", "RW"],
        }
        final_data[species].append(species_data)

    # Create a figure with 2 subplots (one for each species)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Create a 2x3 grid of subplots
    axes = axes.flatten()
    colors = ["red", "green", "blue", "orange", "purple"]

    for idx, species in enumerate(species_list):
        ax = axes[idx]  # Get the subplot axis for the current species

        for i in range(len(final_data[species][0]["method"])):
            create_plot(
                ax,
                final_data[species][0]["fpr"][i],
                final_data[species][0]["tpr"][i],
                final_data[species][0]["roc"][i],
                final_data[species][0]["method"][i],
                colors[i]
            )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve for {species.capitalize()}")
        ax.legend(loc="lower right")

    axes[5].set_visible(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
