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
    fpr, tpr, threshold = roc_curve(
        y, scores, pos_label=1
    )
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, threshold, roc_auc

# def create_plot()


def main():
    print("Generating figures")
    # read in the files and get the score and label
    testing = Path("./results/overlapping_neighbor_data.csv")

    testing_df = read_file(testing)
    results_df = {}

    # calculate AUC values
    fpr, tpr, threshold, roc_auc = get_roc_data(testing_df)
    print(fpr, tpr, threshold, roc_auc)
    # plot the images

    plt.plot(
        fpr,
        tpr,
        color="red",
        lw=2,
        label=f"(area = %0.2f)" % roc_auc,
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
