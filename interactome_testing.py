from pathlib import Path

from tools.helper import create_ppi_network, read_specific_columns


def main():
    print("interactome testing")

    fly_interactome_path = Path("./network/fly_propro.csv")
    fly_go_association_path = Path("./network/fly_proGo.csv")
    zfish_interactome_path = Path("./network/zfish_propro.csv")
    zfish_go_association_path = Path("./network/zfish_proGo.csv")
    bsub_interactome_path = Path("./network/bsub_propro.csv")
    bsub_go_association_path = Path("./network/bsub_proGo.csv")

    interactome_columns = [0, 1]
    interactome = read_specific_columns(bsub_interactome_path, interactome_columns, ",")

    go_inferred_columns = [0, 2]
    go_protein_pairs = read_specific_columns(
        bsub_go_association_path, go_inferred_columns, ","
    )

    # for pair in go_protein_pairs:
    #     print(pair)

    protein_list = []

    # if there is no graph.pickle file in the output/dataset directory, uncomment the following lines
    G, protein_list = create_ppi_network(interactome, go_protein_pairs)

    self_edge_count = 0
    for protein in protein_list:
        print(protein["id"])
        if G.has_edge(protein["id"], protein["id"]):
            self_edge_count+=1
            print("self edge")

    print(self_edge_count)

    # fly has no pro pro self edge
    # zfish has 31 pro pro self edges
    # bsub has 278 pro pro self edges

if __name__ == "__main__":
    main()
