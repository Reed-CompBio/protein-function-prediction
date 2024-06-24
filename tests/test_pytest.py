# tests/test_pytest.py
import pytest
from classes.overlapping_neighbors_class import OverlappingNeighbors
from classes.overlapping_neighbors_v2_class import OverlappingNeighborsV2
from classes.overlapping_neighbors_v3_class import OverlappingNeighborsV3
from classes.protein_degree_class import ProteinDegree
from classes.protein_degree_v2_class import ProteinDegreeV2
from classes.protein_degree_v3_class import ProteinDegreeV3
from classes.sample_algorithm import SampleAlgorithm
from classes.base_algorithm_class import BaseAlgorithm
from classes.hypergeometric_distribution_class import HypergeometricDistribution
from classes.hypergeometric_distribution_class_V2 import HypergeometricDistributionV2

from pathlib import Path
from tools.workflow import run_experiement
from tools.workflow import run_workflow
import os
import pandas as pd


def test_algorithm_attributes():
    algorithm_classes = {
        "OverlappingNeighbors": OverlappingNeighbors,
        "OverlappingNeighborsV2": OverlappingNeighborsV2,
        "OverlappingNeighborsV3": OverlappingNeighborsV3,
        "ProteinDegree": ProteinDegree,
        "ProteinDegreeV2": ProteinDegreeV2,
        "ProteinDegreeV3": ProteinDegreeV3,
        "HypergeometricDistribution": HypergeometricDistribution,
        "HypergeometricDistributionV2": HypergeometricDistributionV2,
    }
    for algorithm in algorithm_classes:
        assert hasattr(algorithm_classes[algorithm](), "y_score")
        assert hasattr(algorithm_classes[algorithm](), "y_true")


def test_algorithm_inherits_class():
    algorithm_classes = {
        "OverlappingNeighbors": OverlappingNeighbors,
        "OverlappingNeighborsV2": OverlappingNeighborsV2,
        "OverlappingNeighborsV3": OverlappingNeighborsV3,
        "ProteinDegree": ProteinDegree,
        "ProteinDegreeV2": ProteinDegreeV2,
        "ProteinDegreeV3": ProteinDegreeV3,
        "HypergeometricDistribution": HypergeometricDistribution,
        "HypergeometricDistributionV2": HypergeometricDistributionV2,
    }

    for algorithm in algorithm_classes:
        assert issubclass(algorithm_classes[algorithm], BaseAlgorithm)


def test_algorithm_experiment():
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("output/dataset"):
        os.makedirs("output/dataset")
    if not os.path.exists("output/data"):
        os.makedirs("output/data")
    if not os.path.exists("output/images"):
        os.makedirs("output/images")

    output_data_path = Path("./output/data/")
    output_image_path = Path("./output/images/")
    input_directory_path = Path("./tests/testing-dataset/")
    graph_file_path = Path(input_directory_path, "graph.pickle")

    algorithm_classes = {
        "OverlappingNeighbors": OverlappingNeighbors,
        "OverlappingNeighborsV2": OverlappingNeighborsV2,
        "OverlappingNeighborsV3": OverlappingNeighborsV3,
        "ProteinDegree": ProteinDegree,
        "ProteinDegreeV2": ProteinDegreeV2,
        "ProteinDegreeV3": ProteinDegreeV3,
        "HypergeometricDistribution": HypergeometricDistribution,
        "HypergeometricDistributionV2": HypergeometricDistributionV2,
    }

    results = run_experiement(
        algorithm_classes,
        input_directory_path,
        graph_file_path,
        output_data_path,
        output_image_path,
        False,
        False,
        0,
        "_mol_bio_cel",
    )
    roc_results = {
        "OverlappingNeighbors": 0.5,
        "OverlappingNeighborsV2": 0.6799999999999999,
        "OverlappingNeighborsV3": 0.6,
        "ProteinDegree": 0.86,
        "ProteinDegreeV2": 0.635,
        "ProteinDegreeV3": 0.855,
        "HypergeometricDistribution": 0.6599999999999999,
        "HypergeometricDistributionV2": 0.72,
    }

    pr_results = {
        "ProteinDegreeV3": 0.852965367965368,
        "ProteinDegree": 0.8828661616161616,
        "OverlappingNeighborsV3": 0.6363911114685108,
        "OverlappingNeighborsV2": 0.7768889469663462,
        "ProteinDegreeV2": 0.7558547008547009,
        "OverlappingNeighbors": 0.5636528325905261,
        "HypergeometricDistribution": 0.7131199315390491,
        "HypergeometricDistributionV2": 0.7647093837535013,
    }
    
    for algorithm, metrics in results.items():
        assert metrics["roc_auc"] == roc_results[algorithm]

    for algorithm, metrics in results.items():
        assert metrics["pr_auc"] == pr_results[algorithm]

def test_multiple_input_files():
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("output/dataset"):
        os.makedirs("output/dataset")
    if not os.path.exists("output/data"):
        os.makedirs("output/data")
    if not os.path.exists("output/images"):
        os.makedirs("output/images")

    output_data_path = Path("./output/data/")
    output_image_path = Path("./output/images/")
    input_directory_path = Path("./tests/testing-dataset/")
    graph_file_path = Path(input_directory_path, "graph.pickle")

    algorithm_classes = {
        "OverlappingNeighbors": OverlappingNeighbors,
        "OverlappingNeighborsV2": OverlappingNeighborsV2,
        "OverlappingNeighborsV3": OverlappingNeighborsV3,
        "ProteinDegree": ProteinDegree,
        "ProteinDegreeV2": ProteinDegreeV2,
        "ProteinDegreeV3": ProteinDegreeV3,
        "HypergeometricDistribution": HypergeometricDistribution,
        "HypergeometricDistributionV2": HypergeometricDistributionV2,
    }
    
    run_workflow(
        algorithm_classes,
        "N/A",
        10,
        "N/A",
        graph_file_path,
        input_directory_path,
        output_data_path,
        output_image_path,
        5,
        False,
        "_mol_bio_cel",
        False,
    )
    roc_mean_results = {
        "OverlappingNeighbors": 0.640,
        "OverlappingNeighborsV2": 0.718,
        "OverlappingNeighborsV3": 0.696,
        "ProteinDegree": 0.803,
        "ProteinDegreeV2": 0.671,
        "ProteinDegreeV3": 0.839,
        "HypergeometricDistribution": 0.672,
        "HypergeometricDistributionV2": 0.862,
    }
    roc_sd_results = {
        "OverlappingNeighbors": 0.13285,
        "OverlappingNeighborsV2": 0.04087,
        "OverlappingNeighborsV3": 0.06786,
        "ProteinDegree": 0.06068,
        "ProteinDegreeV2": 0.04068,
        "ProteinDegreeV3": 0.06749,
        "HypergeometricDistribution": 0.01643,
        "HypergeometricDistributionV2": 0.09985,
    }

    pr_mean_results = {
        "OverlappingNeighbors": 0.63917,
        "OverlappingNeighborsV2": 0.74472,
        "OverlappingNeighborsV3": 0.69077,
        "ProteinDegree": 0.81523,
        "ProteinDegreeV2": 0.69686,
        "ProteinDegreeV3": 0.83840,
        "HypergeometricDistribution": 0.70979,
        "HypergeometricDistributionV2": 0.84226,
    }

    pr_sd_results = {
        "OverlappingNeighbors": 0.11003,
        "OverlappingNeighborsV2": 0.06251,
        "OverlappingNeighborsV3": 0.08707,
        "ProteinDegree": 0.05213,
        "ProteinDegreeV2": 0.04951,
        "ProteinDegreeV3": 0.06645,
        "HypergeometricDistribution": 0.08328,
        "HypergeometricDistributionV2": 0.12172,
    }

    df = pd.read_csv("./output/data/5_repeated_auc_values.csv", sep = "\t").to_dict()

    output = {}
    c = 0
    for i in algorithm_classes.keys():
        output[i] = [df['ROC mean'][c], df['ROC sd'][c], df['Precision/Recall mean'][c], df['Precision/Recall sd'][c]]
        c += 1

    for algorithm in output:
        assert output[algorithm][0] == roc_mean_results[algorithm]
        assert output[algorithm][1] == roc_sd_results[algorithm]
        assert output[algorithm][2] == pr_mean_results[algorithm]
        assert output[algorithm][3] == pr_sd_results[algorithm]
