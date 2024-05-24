from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tools.helper import print_progress, normalize


def run_workflow(
    algorithm_classes, positive_dataset, negative_dataset, G, output_data_path
):
    print("")
    print("-" * 65)
    print("Calculating Protein Prediction")
    results = {}
    i = 1
    for algorithm_name, algorithm_class in algorithm_classes.items():
        print("")
        print(f"{i} / {len(algorithm_classes)}: {algorithm_name} algorithm")
        current = run_algorithm(
            algorithm_class, positive_dataset, negative_dataset, G, output_data_path
        )
        current = run_metrics(current)
        results[algorithm_name] = current
        i+=1
    return results


def run_algorithm(
    algorithm_class,
    positive_dataset,
    negative_dataset,
    G,
    output_data_path,
):
    # Create an instance of the algorithm class
    algorithm = algorithm_class()

    # Predict using the algorithm
    algorithm.predict(positive_dataset, negative_dataset, G, output_data_path)

    # Access y_true and y_score attributes for evaluation
    y_true = algorithm.y_true
    y_score = algorithm.y_score

    results = {"y_true": y_true, "y_score": y_score}

    return results


def run_metrics(current):
    # Compute ROC curve and ROC area for a classifier
    current["fpr"], current["tpr"], current["thresholds"] = roc_curve(
        current["y_true"], current["y_score"]
    )
    current["roc_auc"] = auc(current["fpr"], current["tpr"])

    # Compute precision-recall curve and area under the curve for a classifier
    current["precision"], current["recall"], _ = precision_recall_curve(
        current["y_true"], current["y_score"]
    )
    current["pr_auc"] = auc(current["recall"], current["precision"])

    return current

