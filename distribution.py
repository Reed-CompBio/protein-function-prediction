from tools.helper import (
    read_specific_columns,
    import_graph_from_pickle,
    get_neighbors,
    print_progress,
)
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import os

'''
Creates a series of graphs that visualize the frequency of GO term neighbor counts. The first graph shows the number of go terms that have 0-10 neighbors, the second graph shows the number of go terms that have 0-100 neighbors, the third graph shows go terms that have 100-1000 neighbors, and the third graph shows everything on the same graph. All of the graphs have different X and Y ranges and should not be directly compared to each other. 

'''


# Requires a graph to be pre-created 
G = import_graph_from_pickle("./output/dataset/graph.pickle")
go_terms = read_specific_columns("./network/go_depth.csv", [0], ",")

proteins = {}
for go in go_terms:
    go = go[0]
    num = get_neighbors(G, go, "protein_go_term")
    proteins[go] = len(num)

lst = [] # A list of frequencies for go term annotation
for i in proteins.keys():
    lst.append(proteins[i])
lst = sorted(lst) 
counts = Counter(lst) # The number of times each frequency appears in the list 
lst_range = range(lst[-1])

#Some of this code was pulled from stack overflow
count_dict = {} #Creates a dictionary with a key from 0 to the number of proteins annotated to the largest GO term
for i in lst_range:
    count_dict[i] = 0
for i in counts:
    count_dict[i] = counts[i]

count_dict = pd.Series(count_dict)
x_values = count_dict.index

#Each of these has a different y-range for better visualization
plt.bar(x_values, count_dict.values)
plt.title("Number of GO terms with under 10 protein neighbors")
plt.ylim(0,35000)
plt.xlim(0,10)
plt.show()

plt.bar(x_values, count_dict.values)
plt.title("Number of GO terms with under 100 protein neighbors")
plt.ylim(0,800)
plt.xlim(0,100)
plt.show()

plt.bar(x_values, count_dict.values)
plt.title("Number of GO terms with over 100 protein neighbors")
plt.ylim(0,10)
plt.xlim(100,1000)
plt.show()

plt.bar(x_values, count_dict.values)
plt.title("All together (with 0 and 1 out of y range)")
plt.ylim(0,1000)
plt.xlim(0,1000)
plt.show()