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
from classes.random_walk_class import RandomWalk
from classes.random_walk_class_v2 import RandomWalkV2
from classes.random_walk_class_v3 import RandomWalkV3
from classes.random_walk_class_v4 import RandomWalkV4
from classes.random_walk_class_v5 import RandomWalkV5

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
        "RandomWalk": RandomWalk,
        "RandomWalkV2": RandomWalkV2,
        "RandomWalkV3": RandomWalkV3, 
        "RandomWalkV4": RandomWalkV4,
        "RandomWalkV5": RandomWalkV5,
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
        "OverlappingNeighbors": 0.61,
        "OverlappingNeighborsV2": 0.725,
        "OverlappingNeighborsV3": 0.745,
        "ProteinDegree": 0.7,
        "ProteinDegreeV2": 0.49000000000000005,
        "ProteinDegreeV3": 0.8099999999999999,
        "HypergeometricDistribution": 0.81,
        "HypergeometricDistributionV2": 0.9199999999999999,
        "RandomWalk": 0.99,
        "RandomWalkV2": 0.87,
        "RandomWalkV3": 0.88,
        "RandomWalkV4": 0.71,
        "RandomWalkV5": 0.93,
    }

    pr_results = {
        "OverlappingNeighbors": 0.7155291006722895,
        "OverlappingNeighborsV2": 0.6146794178044178,
        "OverlappingNeighborsV3": 0.6313659257409258,
        "ProteinDegree": 0.720191058941059,
        "ProteinDegreeV2": 0.4746853146853147,
        "ProteinDegreeV3": 0.7830150405150405,
        "HypergeometricDistribution": 0.83235581412052,
        "HypergeometricDistributionV2": 0.9287140637140637,
        "RandomWalk": 0.9904545454545454,
        "RandomWalkV2": 0.8913956876456876,
        "RandomWalkV3": 0.89892094017094,
        "RandomWalkV4": 0.7505560166957226,
        "RandomWalkV5": 0.9228102453102452,
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
        "RandomWalk": RandomWalk,
        "RandomWalkV2": RandomWalkV2,
        "RandomWalkV3": RandomWalkV3,
        "RandomWalkV4": RandomWalkV4,
        "RandomWalkV5": RandomWalkV5,
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
        "OverlappingNeighbors": 0.698,
        "OverlappingNeighborsV2": 0.806,
        "OverlappingNeighborsV3": 0.796,
        "ProteinDegree": 0.76,
        "ProteinDegreeV2": 0.619,
        "ProteinDegreeV3": 0.759,
        "HypergeometricDistribution": 0.788,
        "HypergeometricDistributionV2": 0.934,
        "RandomWalk": 0.988,
        "RandomWalkV2": 0.91,
        "RandomWalkV3": 0.914,
        "RandomWalkV4": 0.77,
        "RandomWalkV5": 0.924,
    }

    roc_sd_results = {
        "OverlappingNeighbors": 0.06419,
        "OverlappingNeighborsV2": 0.08466,
        "OverlappingNeighborsV3": 0.07561,
        "ProteinDegree": 0.09247,
        "ProteinDegreeV2": 0.14935,
        "ProteinDegreeV3": 0.07619,
        "HypergeometricDistribution": 0.14167,
        "HypergeometricDistributionV2": 0.0493,
        "RandomWalk": 0.00837,
        "RandomWalkV2": 0.08573,
        "RandomWalkV3": 0.08081,
        "RandomWalkV4": 0.11247,
        "RandomWalkV5": 0.07987,
    }
    
    pr_mean_results = {
        "OverlappingNeighbors": 0.70429,
        "OverlappingNeighborsV2": 0.79163,
        "OverlappingNeighborsV3": 0.76839,
        "ProteinDegree": 0.76441,
        "ProteinDegreeV2": 0.66936,
        "ProteinDegreeV3": 0.72171,
        "HypergeometricDistribution": 0.81856,
        "HypergeometricDistributionV2": 0.92057,
        "RandomWalk": 0.98868,
        "RandomWalkV2": 0.92106,
        "RandomWalkV3": 0.92387,
        "RandomWalkV4": 0.82391,
        "RandomWalkV5": 0.92862,
    }

    pr_sd_results = {
        "OverlappingNeighbors": 0.07198,
        "OverlappingNeighborsV2": 0.12493,
        "OverlappingNeighborsV3": 0.12056,
        "ProteinDegree": 0.08261,
        "ProteinDegreeV2": 0.1476,
        "ProteinDegreeV3": 0.06049,
        "HypergeometricDistribution": 0.06292,
        "HypergeometricDistributionV2": 0.07011,
        "RandomWalk": 0.00789,
        "RandomWalkV2": 0.07642,
        "RandomWalkV3": 0.07319,
        "RandomWalkV4": 0.08392,
        "RandomWalkV5": 0.06955,
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
