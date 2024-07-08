from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tools.helper import print_progress
from sklearn.metrics import f1_score
from colorama import Fore, Style
import numpy as np
from tools.helper import (
    add_print_statements,
    generate_random_colors,
    import_graph_from_pickle,
)
from pathlib import Path
import matplotlib.pyplot as plt
import random
from random import sample
import pandas as pd
from operator import itemgetter
import statistics as stat
import os
import sys


def run_workflow(
    algorithm_classes,
    go_protein_pairs,
    sample_size,
    protein_list,
    graph_file_path,
    dataset_directory_path,
    output_data_path,
    output_image_path,
    repeats,
    new_random_lists,
    name,
    figure,
):
    """
    With a given set of algorithms, test the algorithms ability to prediction protein function on a given number of
    repetitions and sample size. This starts the workflow to calculate PR and ROC curves

    Parameters:
    algorithm_classes {dict} : a dictionary with keys as algorithm names and values as those algorithms' respective classes
    go_protein_pairs {list} : a list containing the edge between a protein and a go-term e.g. [[protein1, go_term1], [protein2, go_term2], ...]
    sample_size {int} : the size of a positive/negative dataset to be sampled
    protein_list {list} : a list of all proteins in the graph
    graph_file_path {Path} : path of the exported nx graph
    dataset_directory_path {Path} : path of the directory containing the datasets
    output_data_path {Path} : path of the output data
    output_image_path {Path} : path of the output image
    repeats {int} : the number of experiment repetitions
    new_random_list {bool} : flag True to generate completely new pos/neg lists, False to use pre-existing ones 
    name {str} : a string of namespaces chosen to be used in the sample
    figure {bool} : true if graphs should be printed (any), false if not

    Returns:
    Null
    """
    G = import_graph_from_pickle(graph_file_path)
    x = repeats  # Number of replicates
    print_graphs = figure
    if x > 1:
        print_graphs = False
    auc = {}
    # index 0 is ROC, index 1 is Precision Recall
    for i in algorithm_classes.keys():
        auc[i] = [[], []]

    #Sorts through replicates in directory and returns number of dataset pairs, needs to be formatted and ordered corectly
    if new_random_lists == False:
        x = use_existing_samples(dataset_directory_path)

    #Generates completely new positive and negative lists for every replicate, regardless of if the file already exists or not
    else:
        remove_samples(x, dataset_directory_path)
        for i in range(x):
            positive_dataset, negative_dataset = sample_data(
                go_protein_pairs, sample_size, protein_list, G, dataset_directory_path, i, name
            )

    
    for i in range(
        x
    ):  # Creates a pos/neg list each replicate then runs workflow like normal
        if x > 1:
            print("\n\nReplicate: " + str(i) + "\n")

        # positive_dataset, negative_dataset = sample_data(
        #     go_protein_pairs, sample_size, protein_list, G, dataset_directory_path
        # )

        results = run_experiement(
            algorithm_classes,
            dataset_directory_path,
            graph_file_path,
            output_data_path,
            output_image_path,
            True,
            print_graphs,
            i,
            name,
        )
        
        # each loop adds the roc and pr values, index 0 for roc and 1 for pr, for each algorithm
        for i in algorithm_classes.keys():
            auc[i][0].append(results[i]["roc_auc"]) #round(results[i]["roc_auc"],5))
            auc[i][1].append(results[i]["pr_auc"]) #round(results[i]["pr_auc"],5))

    #Creates a dictionary for all pr values and all roc values 
    roc = {}
    pr = {}

    for i in algorithm_classes.keys():
        roc[i] = auc[i][0]
        pr[i] = auc[i][1]
    
    if x > 1:
        cols = []
        for i in range(x):
            cols.append("Replicate " + str(i))
        
        # Finds mean and sd of values, ROC mean index 0, ROC sd index 1, PR mean index 2, and PR sd index 3
        for i in auc.keys():
            meanROC = round(stat.mean(auc[i][0]), 5)
            auc[i].append(round(stat.mean(auc[i][1]), 5))
            auc[i].append(round(stat.stdev(auc[i][1]), 5))
            auc[i][1] = round(stat.stdev(auc[i][0]), 5)
            auc[i][0] = meanROC

        # Prints the roc and pr table, then saves to .csv file
        df = pd.DataFrame.from_dict(
            auc,
            orient="index",
            columns=[
                "ROC mean",
                "ROC sd",
                "Precision/Recall mean",
                "Precision/Recall sd",
            ],
        )
        print()
        print(df)
        df.to_csv(
            Path(output_data_path, str(x) + "_repeated_auc_values.csv"),
            index=True,
            sep="\t",
        )
    else:
        cols = ["AUC"]
        
    dfr = pd.DataFrame.from_dict(
        roc,
        orient = 'index',
        columns = cols
    )

    dfp = pd.DataFrame.from_dict(
        pr,
        orient = 'index',
        columns = cols
    )
        
    dfr.to_csv(
        Path(output_data_path, "roc_auc_results.csv"),
        index = True,
        sep = "\t"
    )
    
    dfp.to_csv(
        Path(output_data_path, "pr_auc_results.csv"),
        index = True,
        sep = "\t"
    )
    if x > 1 & figure == True:
        replicate_boxplot(roc, output_image_path, True)
        replicate_boxplot(pr, output_image_path, False)

def run_experiement(
    algorithm_classes,
    input_directory_path,
    graph_file_path,
    output_data_path,
    output_image_path,
    threshold,
    figures,
    rep_num,
    name,
):
    """
    Run an iteration with a sample dataset on all the algorithms, calculating their protein prediction scores

    Parameters:
    algorithm_classes {dict} : a dictionary with keys as algorithm names and values as those algorithms' respective classes
    input_directory_path {Path} : path of positive and negative datasets
    graph_file_path {Path} : path of the exported nx graph
    output_data_path {Path} : path of the output data
    output_image_path {Path} : path of the output image
    rep_num {int} : replicate number to use associated pos/neg dataset
    name {str} : namespaces used to create the sample datasets

    Returns:
    Results {dictionary} : contains a key value pair where each association algorithms is a key and their values are the metrics and threshold results
    """
    print("")
    print("-" * 65)
    print("Calculating Protein Prediction")
    results = {}
    i = 1
    for algorithm_name, algorithm_class in algorithm_classes.items():
        print("")
        print(f"{i} / {len(algorithm_classes)}: {algorithm_name} Algorithm")
        current = run_algorithm(
            algorithm_class, input_directory_path, graph_file_path, output_data_path, rep_num, name,
        )
        current = run_metrics(current)
        results[algorithm_name] = current
        i += 1

    if threshold:
        run_thresholds(results, algorithm_classes, output_data_path)
        if figures:
            generate_figures(
                algorithm_classes, results, output_image_path, output_data_path
            )

    return results


def run_algorithm(
    algorithm_class,
    input_directory_path,
    graph_file_path,
    output_data_path,
    rep_num,
    name,
):
    """
    With a given dataset, run an algorithm's predict method.

    Parameters:
    algorithm_class {class} : the respective algorithm's class
    input_directory_path {Path} : path of positive and negative datasets
    graph_file_path {Path} : path of the exported nx graph
    output_data_path {Path} : path of the output data
    rep_num {int} : replicate number to use associated pos/neg dataset
    name {str} : namespaces used to create the sample datasets
    

    Returns:
    Result {dict} : a dictionary that stores the y_true and y_score values of the algorithm
    """
    # Create an instance of the algorithm class
    algorithm = algorithm_class()

    # Predict using the algorithm
    y_score, y_true = algorithm.predict(
        input_directory_path, graph_file_path, output_data_path, rep_num, name,
    )

    # Access y_true and y_score attributes for evaluation
    algorithm.set_y_score(y_score)
    algorithm.set_y_true(y_true)

    results = {"y_true": y_true, "y_score": y_score}

    return results


def run_metrics(current):
    """
    Add more keys to the current {dict} that contains metrics and stats

    Parameters:
    current {dict}: a dictionary containing initially the y_score and y_true values of an algorithm

    Returns:
    current {dict} : a dictionary that stores the y_true, y_score, and metrics of a given algorithm.
    """
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


def run_thresholds(results, algorithm_classes, output_data_path):
    """
    Across all the algorithms, calculate their thresholds using three threshold metrics, youden_j, max_f1, and optimal_distance

    Parameters:
    results {dict}: a dictionary that stores the y_true, y_score, and metrics of all the algorithms
    algorithm_classes {dict} : a dictionary containing the algorithms and its respective algorithm class
    output_data_path {Path} : path of the output data

    Returns:
    Null
    """
    print("")
    print("-" * 65)
    print("Calculating Optimal Thresholds")

    j = 1
    threshold_results = []
    # Calculate thresholding for each method w/ three threshold metrics
    for algorithm_name, metrics in results.items():
        print("")
        print(f"{j} / {len(algorithm_classes)}: {algorithm_name} Algorithm")
        # print(f"Calculating optimal thresholds: {algorithm_name}")
        # 1. Maximize the Youden’s J Statistic
        youden_j = metrics["tpr"] - metrics["fpr"]
        optimal_index_youden = np.argmax(youden_j)
        optimal_threshold_youden = metrics["thresholds"][optimal_index_youden]

        # i = 1
        # # 2. Maximize the F1 Score
        # # For each threshold, compute the F1 score
        # f1_scores = []
        # for threshold in metrics["thresholds"]:
        #     y_pred = (metrics["y_score"] >= threshold).astype(int)
        #     f1 = f1_score(metrics["y_true"], y_pred)
        #     f1_scores.append(f1)
        #     print_progress(i, len(metrics["thresholds"]))
        #     i += 1
        # optimal_index_f1 = np.argmax(f1_scores)
        # optimal_threshold_f1 = metrics["thresholds"][optimal_index_f1]

        # 3. Minimize the Distance to (0, 1) on the ROC Curve
        distances = np.sqrt((1 - metrics["tpr"]) ** 2 + metrics["fpr"] ** 2)
        optimal_index_distance = np.argmin(distances)
        optimal_threshold_distance = metrics["thresholds"][optimal_index_distance]

        threshold_results.append(algorithm_name)
        threshold_results.append(
            f"Optimal Threshold (Youden's J): {optimal_threshold_youden}"
        )
        # threshold_results.append(
        #     f"Optimal Threshold (F1 Score): {optimal_threshold_f1}"
        # )
        threshold_results.append(
            f"Optimal Threshold (Min Distance to (0,1)): {optimal_threshold_distance}"
        )

        j += 1

    add_print_statements(
        Path(output_data_path, "threshold_results.txt"), threshold_results
    )


def generate_figures(algorithm_classes, results, output_image_path, output_data_path):
    """
    Generate ROC and PR figures to compare methods

    Parameters:
    algorithm_classes {dict} : a dictionary containing the algorithms and its respective algorithm class
    results {dict}: a dictionary that stores the y_true, y_score, and metrics of all the algorithms
    output_image_path {Path} : path of the output image
    output_data_path {Path} : path of the output data

    Returns:
    Null
    """
    # Generate ROC and PR figures to compare methods

    colors = generate_random_colors(len(algorithm_classes))

    sorted_results = sort_results_by(results, "roc_auc", output_data_path)
    # Initialize your parameters
    fig_width = 10  # width in inches
    fig_height = 7  # height in inches
    fig_dpi = 100  # dots per inch for the figure
    save_dpi = 200  # dots per inch for the saved image

    plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)

    i = 0  # Initialize your index for colors
    for algorithm_name, metrics in sorted_results.items():
        plt.plot(
            metrics["fpr"],
            metrics["tpr"],
            color=colors[i],
            lw=2,
            label=f"{algorithm_name} (area = %0.2f)" % metrics["roc_auc"],
        )
        i += 1

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(Path(output_image_path, "multiple_roc_curves.png"))
    plt.show()

    sorted_results = sort_results_by(results, "pr_auc", output_data_path)
    i = 0
    plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
    for algorithm_name, metrics in sorted_results.items():
        plt.plot(
            metrics["recall"],
            metrics["precision"],
            color=colors[i],
            lw=2,
            label=f"{algorithm_name} (area = %0.2f)" % metrics["pr_auc"],
        )
        i += 1
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower right")
    plt.savefig(Path(output_image_path, "multiple_pr_curves.png"))
    plt.show()


def sample_data(go_protein_pairs, sample_size, protein_list, G, input_directory_path, num, name):
    """
    Given a sample size, generate positive nad negative datasets.

    Parameters:

    go_protein_pairs {list} : a list containing the edge between a protein and a go-term e.g. [[protein1, go_term1], [protein2, go_term2], ...]
    sample_size {int} : the size of a positive/negative dataset to be sampled
    protein_list {list} : a list of all proteins in the graph
    G {nx.Graph} : graph that represents the interactome and go term connections
    input_directory_path {Path} : Path to directory of the datasets
    num {int} : Number of positive/negative dataset
    name {str} : shorthand for all namespaces used to generate datasets, adds shorthand to .csv name

    Returns:
    positive_dataset, negative_dataset

    """
    positive_dataset = {"protein": [], "go": []}
    negative_dataset = {"protein": [], "go": []}
    # sample the data
    for edge in sample(list(go_protein_pairs), sample_size):
        positive_dataset["protein"].append(edge[0])
        positive_dataset["go"].append(edge[1])

    i = 1

    for protein, go in zip(positive_dataset["protein"], positive_dataset["go"]):
        sample_edge = random.choice(protein_list)
        # removes if a protein has a corresponding edge to the GO term in the network
        while G.has_edge(sample_edge["id"], go):
            sample_edge = random.choice(protein_list)
        negative_dataset["protein"].append(sample_edge["id"])
        negative_dataset["go"].append(go)
        print_progress(i, sample_size)
        i += 1

    positive_df = pd.DataFrame(positive_dataset)
    negative_df = pd.DataFrame(negative_dataset)
    
    positive_df.to_csv(
        Path(input_directory_path, "rep_" + str(num) + "_positive_protein_go_term_pairs" + name + ".csv"),
        index=False,
        sep="\t",
    )
    negative_df.to_csv(
        Path(input_directory_path, "rep_" + str(num) + "_negative_protein_go_term_pairs" + name + ".csv"),
        index=False,
        sep="\t",
    )

    return positive_dataset, negative_dataset


def get_datasets(input_directory_path, rep_num, name):
    """
    get the positive and negative datasets as lists by reading their csv files

    Parameters:

    input_directory_path {Path} : Path to directory of the datasets
    rep_num {int} : Replicate number to specify which positive and negative list to use
    name {str} : string of namespaces contained in the .csv file name

    Returns:
    positive_dataset, negative_dataset

    """
    positive_dataset = {"protein": [], "go": []}
    negative_dataset = {"protein": [], "go": []}
    try:
        with open(
            Path(input_directory_path, "rep_" + str(rep_num) + "_positive_protein_go_term_pairs" + name + ".csv"), "r"
        ) as file:
            next(file)
            for line in file:
                parts = line.strip().split("\t")
                # print(parts[0])
                positive_dataset["protein"].append(parts[0])
                positive_dataset["go"].append(parts[1])
    except:
        print(Fore.RED + "\nFile not found: ensure given namespaces match positive and negative file namespaces\n")
        sys.exit(1)

    with open(
        Path(input_directory_path, "rep_" + str(rep_num) + "_negative_protein_go_term_pairs" + name + ".csv"), "r"
    ) as file:
        next(file)
        for line in file:
            parts = line.strip().split("\t")
            # print(parts[0])
            negative_dataset["protein"].append(parts[0])
            negative_dataset["go"].append(parts[1])

    return positive_dataset, negative_dataset


def sort_results_by(results, key, output_path):
    """
    Given a the results, sort them by value of ROC/PR

    Parameters:

    results {dict}: a dictionary that stores the y_true, y_score, and metrics of all the algorithms
    key {str} : PR or ROC
    output_path {Path} : Path to where to write the pr/roc results

    Returns:
    sorted_results {dict }

    """
    algorithm_tuple_list = []
    data = {"algorithm": [], key: []}
    output_file_path = Path(output_path, key + "_results.csv")

    # make a list of tuples where a tuple is (algorithm_name, the metric we will be sorting by)
    for algorithm_name, metrics in results.items():
        algorithm_tuple_list.append((algorithm_name, metrics[key]))
        data["algorithm"].append(algorithm_name)
        data[key].append(metrics[key])

    df = pd.DataFrame(data)
    df = df.sort_values(by=key, ascending=False)

    # df.to_csv(
    #     output_file_path,
    #     index=False,
    #     sep="\t",
    # )

    algorithm_tuple_list = sorted(algorithm_tuple_list, key=itemgetter(1), reverse=True)

    sorted_results = {}
    for algorithm in algorithm_tuple_list:
        sorted_results[algorithm[0]] = results[algorithm[0]]
    return sorted_results

def remove_samples(x, dataset_directory_path):
    """
    Removes all old samples before creating new ones (to ensure no issues when changing namespaces)

    Parameters:

    x {int}: the number of replicates
    dataset_directory_path {Path}: path to the dataset directory where samples are stored
    
    Returns:
    Null

    """
    data_dir = sorted(os.listdir(dataset_directory_path))
    remove = [] 
    files = {'positive': {},
       'negative': {}
        } 
    for i in data_dir:
        temp = i.split("_")
        if temp[0] == 'rep':
            rep_num = int(temp[1])
            if temp[2] == 'positive':
                files['positive'][rep_num] = i
                remove.append(rep_num)
            elif temp[2] == 'negative':
                files['negative'][rep_num] = i
                
    for i in remove:
        del_file_path_pos = Path(dataset_directory_path, files['positive'][i])
        del_file_path_neg = Path(dataset_directory_path, files['negative'][i])
        os.remove(del_file_path_pos)
        os.remove(del_file_path_neg)


def use_existing_samples(dataset_directory_path):
    """
    Sorts through repititions and renames files to ensure they are 0-x with one step between

    Parameters:

    dataset_directory_path {Path}: path to the dataset directory where samples are stored
    
    Returns:
    the number of repititions in the directory {int}

    """
    data_dir = sorted(os.listdir(dataset_directory_path))
    nums = [] 
    for i in data_dir:
        temp = i.split("_")
        if temp[0] == 'rep':
            rep_num = int(temp[1])
            if temp[2] == 'positive':
                nums.append(rep_num)
    nums = sorted(nums)
    n = len(nums)
    return n

def replicate_boxplot(auc_list, output_image_path, curve):
    """
    Creates a boxplot using replicates of the AUC value for ROC or PR curves

    Parameters:

    auc_list {dict}: either ROC or PR dictionary containing the list of AUC values for each replicate
    output_image_path {Path} : output path to save the graph
    curve {bool} : either True for ROC or False for PR 
    
    Returns:
    NULL

    """
    graph = []
    col_names = ["ON", "ON2", "ON3", "PD", "PD2", "PD3", "SA", "HD", "HD2"]
    colors = ["lightcoral", "indianred", "firebrick", "peachpuff", "sandybrown", "peru", "gold", "goldenrod", "darkgoldenrod", "yellowgreen", "olivedrab", "darkolivegreen", "darkturquoise", "mediumturquoise", "darkcyan", "mediumpurple", "darkviolet", "rebeccapurple", "hotpink", "deeppink", "mediumvioletred"]
    len_keys = len(auc_list.keys())
    ran = random.randrange(len(colors)-len_keys)
    colors = colors[ran:ran+len_keys]
    for i in auc_list:
        graph.append(auc_list[i])
    
    fig, ax = plt.subplots()
    ax.set_ylabel("AUC")
    
    plot = ax.boxplot(graph, 
                      patch_artist = True,
                      labels = col_names)
    
    for patch, color in zip(plot['boxes'], colors):
        patch.set_facecolor(color)
    
    if curve == True:
        plt.title("ROC replicates")
        plt.savefig(Path(output_image_path, "roc_replicate_boxplot.png"))
    else:
        plt.title("PR replicates")
        plt.savefig(Path(output_image_path, "pr_replicate_boxplot.png"))
    plt.show()