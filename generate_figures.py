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


def create_plot(x_data: list, y_data: list, auc: float, type: str) -> None:
    plt.plot(
        x_data,
        y_data,
        color="red",
        lw=2,
        label=f"{type}(area = %0.2f)" % auc,
    )


def main():
    print("Generating figures")
    # read in the files and get the score and label
    species_list = ["elegans"]
    final_data = {}
    for species in species_list:
        overlapping_path = Path(
            "./results/final-inferred-complete/elegans/overlapping_neighbor_data.csv"
        )
        hypergeometric_path = Path(
            "./results/final-inferred-complete/elegans/hypergeometric_distribution.csv"
        )
        degree_path = Path(
            "./results/final-inferred-complete/elegans/protein_degree_v3_data.csv"
        )
        rw_path = Path("./results/final-inferred-complete/elegans/random_walk_data_v2.csv")

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
        final_data.append(species, species_data)

    for species in species_list:
        # # plot the images
        fig_width = 10  # width in inches
        fig_height = 7  # height in inches
        fig_dpi = 100  # dots per inch for the figure
        save_dpi = 200  # dots per inch for the saved image
        plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)

        for i in range(len(final_data[species]["method"])):
            print(i)
            create_plot(
            final_data[species]["fpr"][i],
            final_data[species]["tpr"][i],
            final_data[species]["roc"][i],
            final_data[species]["method"][i],
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()


if __name__ == "__main__":
    main()
