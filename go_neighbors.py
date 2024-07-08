from tools.helper import (
    import_graph_from_pickle,
    #read_pro_go_data,
    read_specific_columns
)
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import os


#The first part of this algorithm does not actually take into account any protein that are not annotated to any go terms

range_higher = 75
range_lower = 25
p = 0
print_graphs = False


# go = read_specific_columns(
#         "./network/fly_proGo.csv", [0], ","
#     )

#Without inferred edges
go = read_specific_columns(
    "./network/gene_association.fb", [1], "\t"
)
print(go)

pro = read_specific_columns(
        "./network/fly_propro.csv", [0,1], ","
    )

proteins = {} # Number of proteins annotated to each go term
for term in pro:
    x = term[0]
    y = term[1]
    if x not in proteins.keys():
        proteins[x] = 0
    if y not in proteins.keys():
        proteins[y] = 0

for term in go:
    i = term[0]
    if i in proteins.keys():
        proteins[i] += 1
    else:
        proteins[i] = 1

lst = [] # A list of frequencies for go term annotation
for i in proteins.keys():
    lst.append(proteins[i])
lst = sorted(lst) 
counts = Counter(lst) # The number of times each frequency appears in the list 
lst_range = range(lst[-1]) #the largest number of go terms annotated to a protein in lst

#Some of this code was pulled from stack overflow
count_dict = {} #Creates a dictionary with a key from 0 to the number of proteins annotated to the largest GO term
for i in lst_range:
    count_dict[i] = 0

over = 0
mid = 0
under = 0
#Adds the number of times the value occurs to the value in the full list
for i in counts:
    count_dict[i] = counts[i]
    if i > range_higher:
        over += counts[i]
    elif i > range_lower:
        mid += counts[i]
    else:
        under += counts[i]
print("Number of proteins with over 300 go term neighbors: " + str(over))
print("Number of proteins with between 100 and 300 go term neighbors: " + str(mid))
print("Number of proteins with under 100 go term neighbors: " + str(under))
print("Total number should be " + str(len(lst)) + ", and is: " + str(over + mid + under) + "\n")

    
count_dict = pd.Series(count_dict)
x_values = count_dict.index

# Each plot has a different y axis for visualization
if print_graphs:
    plt.bar(x_values, count_dict.values)
    plt.title("Number of protins with under 100 go term neighbors")
    plt.ylim(0,1100)
    plt.xlim(-1,100)
    plt.show()
    plt.bar(x_values, count_dict.values)
    plt.title("Number of proteins with between 100 and 300 go term neighbors")
    plt.xlim(100,300)
    plt.ylim(0,50)
    plt.show()
    plt.bar(x_values, count_dict.values)
    plt.title("Number of proteins with over 300 go term neighbors")
    plt.xlim(300, 500)
    plt.ylim(0,5)
    plt.show()  

def neighbors(dataset, terms, range_higher, range_lower, j):
    c = 0
    for i in dataset:
        dataset[c] = i[0]
        c += 1
    
    above_len= 0
    between_len = 0
    below_len = 0
    zero = 0
    
    for i in dataset:
        if i in terms.keys():
            size = terms[i]
            if size > range_higher:
                above_len += 1
            elif size >= range_lower:
                between_len += 1
            else:
                below_len += 1
        else:
            zero += 1            
    return [zero, below_len, between_len, above_len]



reps_pos = {}
reps_neg = {}

data_dir = sorted(os.listdir("./output/dataset"))
replicates = 0
for i in data_dir:
    temp = i.split("_")
    if temp[0] == 'rep' and temp[2] == 'positive':
        replicates += 1
        
for j in range(replicates):
    pos = read_specific_columns("./output/dataset/rep_" + str(j) + "_positive_protein_go_term_pairs_mol_bio_cel.csv", [p], "\t")
    neg = read_specific_columns("./output/dataset/rep_" + str(j) + "_negative_protein_go_term_pairs_mol_bio_cel.csv", [p], "\t")
    
    
    name = "Replicate_" + str(j)
    reps_pos[name] = neighbors(pos, proteins, range_higher, range_lower, j)
    reps_neg[name] = neighbors(neg, proteins, range_higher, range_lower, j)
    # print("Number of Go terms in the positive list with more than 500 neighbors: " + str(above_len))
    # print("Number of Go terms in the positive list with more than 100 and less than 500 neighbors: " + str(between_len))
    # print("Number of Go terms in the positive list with less than 100 neighbors: " + str(below_len))
    # print("Total numbere of Go terms in the positive list should be " + str(len(pos)) + " and is: " + str(above_len + between_len + below_len))

cols = ["No Go term neighbors", "Under " + str(range_lower) + " neighbors", "Between " + str(range_lower) + " and " + str(range_higher) + " neighbors", "Above " + str(range_higher) + " neighbors"]

df_pos = pd.DataFrame.from_dict(
    reps_pos,
    orient="index",
    columns= cols,
)

df_neg = pd.DataFrame.from_dict(
    reps_neg,
    orient = "index",
    columns = cols,
)

print("Positive list of protein neighbors:")
print(df_pos)
print("\n\n\n")
print("Negative list of protein neighbors:")
print(df_neg)