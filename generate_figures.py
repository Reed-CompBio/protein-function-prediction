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

    overlapping_df = read_file(overlapping_path)
    hypergeometric_df = read_file(hypergeometric_path)
    degree_df = read_file(degree_path)
    rw_df = read_file(rw_path)

    # calculate AUC values
    fpr1, tpr1, threshold1, roc_auc1 = get_roc_data(overlapping_df)
    fpr2, tpr2, threshold2, roc_auc2 = get_roc_data(hypergeometric_df)
    fpr3, tpr3, threshold3, roc_auc3 = get_roc_data(degree_df)
    fpr4, tpr4, threshold4, roc_auc4 = get_roc_data(rw_df)

    elegans = {
        "fpr": [fpr1, fpr2, fpr3, fpr4],
        "tpr": [tpr1, tpr2, tpr3, tpr4],
        "roc": [roc_auc1, roc_auc2, roc_auc3, roc_auc4],
        "method": ["Overlapping", "Hypergeometric", "Degree", "RW"],
    }
    final_data = {"elegans": elegans}

    # # plot the images
    fig_width = 10  # width in inches
    fig_height = 7  # height in inches
    fig_dpi = 100  # dots per inch for the figure
    save_dpi = 200  # dots per inch for the saved image
    plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)

    create_plot(
        final_data["elegans"]["fpr"][0],
        final_data["elegans"]["tpr"][0],
        final_data["elegans"]["roc"][0],
        final_data["elegans"]["method"][0],
    )
    create_plot(
        final_data["elegans"]["fpr"][0],
        final_data["elegans"]["tpr"][0],
        final_data["elegans"]["roc"][0],
        final_data["elegans"]["method"][0],
    )
    
    # create_plot(fpr2, tpr2, roc_auc2, "hypergeometric")
    # create_plot(fpr3, tpr3, roc_auc3, "degree")
    # create_plot(fpr4, tpr4, roc_auc4, "rw")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
