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
from pathlib import Path
from tools.helper import (
    read_specific_columns,
    import_graph_from_pickle,
)
from tools.workflow import run_workflow


def test_algorithm_attributes():
    algorithm_classes = {
        "OverlappingNeighbors": OverlappingNeighbors,
        "OverlappingNeighborsV2": OverlappingNeighborsV2,
        "OverlappingNeighborsV3": OverlappingNeighborsV3,
        "ProteinDegree": ProteinDegree,
        "ProteinDegreeV2": ProteinDegreeV2,
        "ProteinDegreeV3": ProteinDegreeV3,
        "SampleAlgorithm": SampleAlgorithm,
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
        "SampleAlgorithm": SampleAlgorithm,
    }

    for algorithm in algorithm_classes:
        assert issubclass(algorithm_classes[algorithm], BaseAlgorithm)


def test_algorithm_workflow():
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
    }

    results = run_workflow(
        algorithm_classes,
        input_directory_path,
        graph_file_path,
        output_data_path,
        output_image_path,
        False,
        False,
    )
    roc_results = {
        "OverlappingNeighbors": 0.6000000000000001,
        "OverlappingNeighborsV2": 0.765,
        "OverlappingNeighborsV3": 0.78,
        "ProteinDegree": 0.825,
        "ProteinDegreeV2": 0.675,
        "ProteinDegreeV3": 0.89,
        "HypergeometricDistribution": 0.78,
        "HypergeometricDistributionV2": 0.89,
        "HypergeometricDistributionV3": 0.675,
        "HypergeometricDistributionV4": 0.6
    }

    pr_results = {
        "ProteinDegreeV3": 0.8459976134976135,
        "ProteinDegree": 0.793134088134088,
        "OverlappingNeighborsV3": 0.7737824675324676,
        "OverlappingNeighborsV2": 0.7467907092907092,
        "ProteinDegreeV2": 0.6367757242757243,
        "OverlappingNeighbors": 0.5329058916229968,
        "SampleAlgorithm": 0.4093791854859966,
        "HypergeometricDistribution": 0.7899246806,
        "HypergeometricDistributionV2": 0.8519169719,
        "HypergeometricDistributionV3": 0.7142573629,
        "HypergeometricDistributionV4": 0.6967847007,
    }

    for algorithm, metrics in results.items():
        assert metrics["roc_auc"] == roc_results[algorithm]

    for algorithm, metrics in results.items():
        assert metrics["pr_auc"] == pr_results[algorithm]
