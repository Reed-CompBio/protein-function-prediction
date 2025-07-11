from collections import defaultdict
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


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


def get_pr_data(data_df: dict) -> list:
    y = np.array(data_df["y_true"])
    scores = np.array(data_df["y_score"])
    precision, recall, _ = precision_recall_curve(y, scores)
    pr_auc = auc(recall, precision)

    return precision, recall, pr_auc


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
    species_list = ["elegans", "fly", "bsub", "yeast", "zfish", "athaliana", "ecoli"]
    species_title = [
        "C. elegans",
        "D. melanogaster",
        "B. subtilis",
        "S. cerevisiae",
        "D. rerio",
        "A. thaliana",
        "E. coli",
    ]

    file_directories = [
        "./results/final-non-inferred-complete/",
        "./results/final-inferred-complete/",
    ]
    subplot_titles = ["Complete Non Inferred Networks", "Complete Inferred Networks"]
    k = 0
    for directory in file_directories:
        final_category_data = defaultdict(list)

        for species in species_list:
            overlapping_path = Path(
                directory, f"{species}/overlapping_neighbor_data.csv"
            )
            hypergeometric_path = Path(
                directory, f"{species}/hypergeometric_distribution.csv"
            )
            degree_path = Path(directory, f"{species}/protein_degree_v3_data.csv")
            rw_path = Path(directory, f"{species}/random_walk_data_v2.csv")

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
            precision_list = []
            recall_list = []
            pr_auc_list = []

            for data in methods:
                fpr, tpr, threshold, roc_auc = get_roc_data(data)
                fpr_list.append(fpr)
                tpr_list.append(tpr)
                threshold_list.append(threshold)
                roc_auc_list.append(roc_auc)

                precision, recall, pr_auc = get_pr_data(data)
                precision_list.append(precision)
                recall_list.append(recall)
                pr_auc_list.append(pr_auc)

            species_data = {
                "fpr": fpr_list,
                "tpr": tpr_list,
                "roc": roc_auc_list,
                "precision": precision_list,
                "recall": recall_list,
                "pr": pr_auc_list,
                "method": ["One-Hop GO Overlap", "Hypergeometric", "Degree", "RW"],
            }
            final_category_data[species].append(species_data)

        # Create a figure with 2 subplots (one for each species)
        fig, axes = plt.subplots(
            2, 4, figsize=(17, 12)
        )  # Create a 2x3 grid of subplots
        axes = axes.flatten()
        colors = ["red", "green", "blue", "orange", "purple", "black", "yellow"]

        for idx, species in enumerate(species_list):
            ax = axes[idx]  # Get the subplot axis for the current species

            for i in range(len(final_category_data[species][0]["method"])):
                create_plot(
                    ax,
                    final_category_data[species][0]["fpr"][i],
                    final_category_data[species][0]["tpr"][i],
                    final_category_data[species][0]["roc"][i],
                    final_category_data[species][0]["method"][i],
                    colors[i],
                )

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate", fontsize=14)
            ax.set_ylabel("True Positive Rate", fontsize=14)
            ax.set_title(f"${species_title[idx].capitalize()}$", fontsize=14)
            ax.legend(loc="lower right", fontsize=12)

        axes[7].set_visible(False)
        fig.suptitle("ROC Curve for All Species w/ " + subplot_titles[k], fontsize=20)
        # Adjust layout to prevent overlap
        plt.tight_layout()  # Adjust rect to accommodate legends
        # Adjust the space between subplots
        plt.subplots_adjust(wspace=0.2)
        plt.savefig(
            Path(
                "./results/images/",
                f"roc_{subplot_titles[k].lower().replace(" ", "_")}.pdf",
            ),
            format="pdf",
        )
        plt.show()

        fig, axes = plt.subplots(
            2, 4, figsize=(17, 12)
        )  # Create a 2x3 grid of subplots
        axes = axes.flatten()
        colors = ["red", "green", "blue", "orange", "purple", "black", "yellow"]

        for idx, species in enumerate(species_list):
            ax = axes[idx]  # Get the subplot axis for the current species

            for i in range(len(final_category_data[species][0]["method"])):
                create_plot(
                    ax,
                    final_category_data[species][0]["recall"][i],
                    final_category_data[species][0]["precision"][i],
                    final_category_data[species][0]["pr"][i],
                    final_category_data[species][0]["method"][i],
                    colors[i],
                )

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Recall", fontsize=14)
            ax.set_ylabel("Precision", fontsize=14)
            ax.set_title(f"${species_title[idx].capitalize()}$", fontsize=14)
            ax.legend(loc="lower right", fontsize=12)

        axes[7].set_visible(False)
        fig.suptitle(
            "Precision/Recall Curve for All Species w/ " + subplot_titles[k],
            fontsize=20,
        )
        # Adjust layout to prevent overlap
        plt.tight_layout()  # Adjust rect to accommodate legends
        # Adjust the space between subplots
        plt.subplots_adjust(wspace=0.2)
        plt.savefig(
            Path(
                "./results/images/",
                f"pr_{subplot_titles[k].lower().replace(" ", "_")}.pdf",
            ),
            format="pdf",
        )
        plt.show()
        k += 1

    # generate RW figures

    species_list = ["elegans", "fly", "bsub", "yeast", "zfish", "athaliana", "ecoli"]
    file_directories = [
        "./results/final-rw-inferred-regular/",
        "./results/final-rw-inferred-pro-go/",
        "./results/final-rw-non-inferred-regular/",
        "./results/final-rw-non-inferred-pro-go/",
    ]
    subplot_titles = [
        "Inferred Complete Network",
        "Inferred ProGo Network",
        "Non Inferred Complete Network",
        "Non Inferred ProGo Network",
    ]
    final_rw_data = defaultdict(list)

    # Load data for each directory and species
    for directory in file_directories:
        for species in species_list:
            rw_path = Path(directory, f"{species}/random_walk_data_v2.csv")
            data = read_file(rw_path)

            # calculate AUC values
            fpr, tpr, threshold, roc_auc = get_roc_data(data)
            precision, recall, pr_auc = get_pr_data(data)

            species_data = {
                "fpr": fpr,
                "tpr": tpr,
                "roc": roc_auc,
                "precision": precision,
                "recall": recall,
                "pr": pr_auc,
            }
            final_rw_data[species].append(species_data)

    # Create a 2x2 subplot layout
    fig, axs = plt.subplots(1, 4, figsize=(40, 12))  # 2 rows, 2 columns
    axs = axs.flatten()  # Flatten to easily index the subplots

    colors = ["red", "green", "blue", "orange", "purple", "black", "magenta"]

    # Plot data for each directory on a subplot
    for idx, directory in enumerate(file_directories):
        ax = axs[idx]  # Get the corresponding subplot
        for i, species in enumerate(species_list):
            ax.plot(
                final_rw_data[species][idx]["fpr"],
                final_rw_data[species][idx]["tpr"],
                color=colors[i],
                lw=2,
                label=f"${species_title[i]}$ (area = %0.2f)"
                % final_rw_data[species][idx]["roc"],
            )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=30)
        ax.set_ylabel("True Positive Rate", fontsize=30)
        ax.set_title(f"{subplot_titles[idx]}", fontsize=30)
        ax.legend(loc="lower right", fontsize=30)
        ax.tick_params(axis="both", which="major", labelsize=20)

    # Adjust layout and show the plot
    fig.suptitle(
        "ROC Curve for RandomWalk Configuration",
        fontsize=45,
    )
    plt.tight_layout()
    plt.savefig(Path("./results/images/rw_roc.pdf"), format="pdf")
    plt.show()

    fig, axs = plt.subplots(1, 4, figsize=(40, 12))  # 2 rows, 2 columns
    axs = axs.flatten()  # Flatten to easily index the subplots

    colors = ["red", "green", "blue", "orange", "purple", "black", "magenta"]

    # Plot data for each directory on a subplot
    for idx, directory in enumerate(file_directories):
        ax = axs[idx]  # Get the corresponding subplot
        for i, species in enumerate(species_list):
            ax.plot(
                final_rw_data[species][idx]["recall"],
                final_rw_data[species][idx]["precision"],
                color=colors[i],
                lw=2,
                label=f"${species_title[i]}$ (area = %0.2f)"
                % final_rw_data[species][idx]["pr"],
            )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=30)
        ax.set_ylabel("Precision", fontsize=30)
        ax.set_title(f"{subplot_titles[idx]}", fontsize=30)
        ax.legend(loc="lower right", fontsize=30)
        ax.tick_params(axis="both", which="major", labelsize=20)

    # Adjust layout and show the plot
    fig.suptitle(
        "Precision/Recall Curve for RandomWalk Configuration",
        fontsize=45,
    )
    plt.tight_layout()
    plt.savefig(Path("./results/images/rw_pr.pdf"), format="pdf")
    plt.show()


if __name__ == "__main__":
    main()
