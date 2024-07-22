from colorama import Fore, Style
import networkx as nx
import random
import numpy as np
import pickle


def print_progress(current, total, bar_length=65):
    # Calculate the progress as a percentage
    percent = float(current) / total
    # Determine the number of hash marks in the progress bar
    arrow = "-" * int(round(percent * bar_length) - 1) + ">"
    spaces = " " * (bar_length - len(arrow))

    # Choose color based on completion
    if current < total:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN

    # Construct the progress bar string
    progress_bar = f"[{arrow + spaces}] {int(round(percent * 100))}%"

    # Print the progress bar with color, overwriting the previous line
    print(f"\r{color}{progress_bar}{Style.RESET_ALL}", end="")


def create_ppi_network(fly_interactome, fly_GO_term, go_depth_dict):
    print("Initializing network")
    i = 1
    total_progress = len(fly_interactome) + len(fly_GO_term)
    G = nx.Graph()
    protein_protein_edge = 0
    protein_go_edge = 0
    protein_node = 0
    go_node = 0
    protein_list = []
    go_term_list = []

    # go through fly interactome, add a new node if it doesnt exists already, then add their physical interactions as edges
    for line in fly_interactome:
        print(line)
        if not G.has_node(line[0]):
            G.add_node(line[0], name=line[0], type="protein")
            protein_list.append({"id": line[0], "name": line[0]})
            protein_node += 1

        if not G.has_node(line[1]):
            G.add_node(line[1], name=line[1], type="protein")
            protein_list.append({"id": line[1], "name": line[1]})
            protein_node += 1

        G.add_edge(line[0], line[1], weight = 1, type="protein_protein")
        protein_protein_edge += 1
        print_progress(i, total_progress)
        i += 1

    # Proteins annotated with a GO term have an edge to a GO term node
    for line in fly_GO_term:
        print(line)
        if not G.has_node(line[1]):
            G.add_node(line[1], type="go_term", weight=go_depth_dict[line[1]])
            go_term_list.append(line[1])
            go_node += 1

        if not G.has_node(line[0]):
            G.add_node(line[0], name=line[0], type="protein")
            protein_list.append({"id": line[0], "name": line[0]})
            protein_node += 1

        G.add_edge(line[1], line[0], weight = go_depth_dict[line[1]], type="protein_go_term")
        protein_go_edge += 1
        print_progress(i, total_progress)
        i += 1

    print("")
    print("")
    print("network summary")

    print("protein-protein edge count: ", protein_protein_edge)
    print("protein-go edge count: ", protein_go_edge)
    print("protein node count: ", protein_node)
    print("go node count: ", go_node)
    print("total edge count: ", len(G.edges()))
    print("total node count: ", len(G.nodes()))

    return G, protein_list
    

def create_only_protein_network(fly_interactome, fly_GO_term):
    print("\nInitializing network")
    i = 1
    total_progress = len(fly_interactome)
    G = nx.Graph()
    protein_protein_edge = 0
    protein_node = 0
    protein_list = []

    # go through fly interactome, add a new node if it doesnt exists already, then add their physical interactions as edges
    for line in fly_interactome:
        if not G.has_node(line[0]):
            G.add_node(line[0], name=line[0], type="protein")
            protein_list.append({"id": line[0], "name": line[0]})
            protein_node += 1

        if not G.has_node(line[1]):
            G.add_node(line[1], name=line[1], type="protein")
            protein_list.append({"id": line[1], "name": line[1]})
            protein_node += 1

        G.add_edge(line[0], line[1], type="protein_protein")
        protein_protein_edge += 1
        print_progress(i, total_progress)
        i += 1
    for line in fly_GO_term:
        if not G.has_node(line[0]):
            G.add_node(line[0], name=line[0], type="protein")
            protein_list.append({"id": line[0], "name": line[0]})
            protein_node += 1
            

    return G #, protein_list

def create_go_protein_only_network(fly_interactome, fly_GO_term):
    print("\nInitializing network")
    i = 1
    total_progress = len(fly_GO_term)
    G = nx.Graph()
    protein_protein_edge = 0
    protein_go_edge = 0
    protein_node = 0
    go_node = 0
    protein_list = []
    go_term_list = []

    # go through fly interactome, add a new node if it doesnt exists already, then add their physical interactions as edges
    for line in fly_interactome:
        if not G.has_node(line[0]):
            G.add_node(line[0], name=line[0], type="protein")
            protein_list.append({"id": line[0], "name": line[0]})
            protein_node += 1

        if not G.has_node(line[1]):
            G.add_node(line[1], name=line[1], type="protein")
            protein_list.append({"id": line[1], "name": line[1]})
            protein_node += 1

        # G.add_edge(line[0], line[1], type="protein_protein")
        # protein_protein_edge += 1
        # print_progress(i, total_progress)
        # i += 1

    # Proteins annotated with a GO term have an edge to a GO term node
    for line in fly_GO_term:
        if not G.has_node(line[1]):
            G.add_node(line[1], type="go_term")
            go_term_list.append(line[1])# 
            go_node += 1

        if not G.has_node(line[0]):
            G.add_node(line[0], name=line[0], type="protein")
            protein_list.append({"id": line[0], "name": line[0]})
            protein_node += 1
            
        G.add_edge(line[1], line[0], type="protein_go_term")
        protein_go_edge += 1
        print_progress(i, total_progress)
        i += 1

    return G


def read_specific_columns(file_path, columns, delimit):
    try:
        with open(file_path, "r") as file:
            next(file)
            data = []
            for line in file:
                parts = line.strip().split(delimit)
                selected_columns = []
                for col in columns:
                    selected_columns.append(parts[col].replace('"', ""))
                data.append(selected_columns)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def read_pro_go_data(file_path, columns, namespace, delimit):
    try:
        with open(file_path, "r") as file:
            next(file)
            data = []
            for line in file:
                parts = line.strip().split(delimit)
                selected_columns = []
                for col in columns:
                    selected_columns.append(parts[col].replace('"', ""))
                if selected_columns[2] in namespace:
                    data.append(selected_columns)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def read_go_depth_data(file_path, columns, namespace, delimit):
    try:
        print("reading go depth")
        with open(file_path, "r") as file:
            next(file)
            data = {}
            for line in file:
                parts = line.strip().split(delimit)
                selected_columns = []
                for col in columns:
                    selected_columns.append(parts[col].replace('"', ""))
                if selected_columns[1] in namespace:
                    data[selected_columns[0]] = selected_columns[2]
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def generate_random_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        color = (random.random(), random.random(), random.random())
        colors.append(color)
    return colors


def normalize(data):
    data = np.array(data)
    min_val = data.min()
    max_val = data.max()

    if min_val == max_val:
        return np.zeros_like(data)

    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data.tolist()


def get_neighbors(G: nx.Graph, node, edgeType):
    res = G.edges(node, data=True)
    neighbors = []
    for edge in res:
        if edge[2]["type"] == edgeType:
            neighborNode = [edge[1], edge[2]]
            neighbors.append(neighborNode)

    return neighbors


def add_print_statements(filename, statements):
    # Open the file in append mode (will create the file if it doesn't exist)
    with open(filename, "w") as file:
        for statement in statements:
            # Write each statement to the file
            file.write(f"{statement}\n")


def export_graph_to_pickle(graph, filename):
    with open(filename, "wb") as f:
        pickle.dump(graph, f)


def import_graph_from_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
