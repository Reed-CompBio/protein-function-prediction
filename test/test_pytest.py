import pytest
import networkx as nx
from pathlib import Path
from tools.helper import read_specific_columns, create_ppi_network
from tools.workflow import sample_data


@pytest.fixture
def generate_ppi():
    interactome_path = Path("./network/interactome-flybase-collapsed-weighted.txt")
    go_association_path = Path("./network/fly_proGo.csv")
    output_data_path = Path("./output/data/")
    output_image_path = Path("./output/images/")
    input_directory_path = Path("./input")
    sample_size = 100


    interactome_columns = [0, 1, 4, 5]
    interactome = read_specific_columns(interactome_path, interactome_columns, "\t")

    go_inferred_columns = [0, 2]
    go_protein_pairs = read_specific_columns(
        go_association_path, go_inferred_columns, ","
    )

    protein_list = []

    G, protein_list = create_ppi_network(interactome, go_protein_pairs)

    positive_dataset, negative_dataset = sample_data(
    go_protein_pairs, sample_size, protein_list, G, input_directory_path
    )
    return G, positive_dataset, negative_dataset

def test_generation_1(generate_ppi):
    G, positive_dataset, negative_dataset = generate_ppi
    print(positive_dataset["protein"][0], negative_dataset["protein"][0])
    assert 0 == 0

def test_generation_2(generate_ppi):
    G, positive_dataset, negative_dataset = generate_ppi
    print(positive_dataset["protein"][0], negative_dataset["protein"][0])

    assert 0 == 0