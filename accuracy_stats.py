import matplotlib.pyplot as plt
import statistics as stat
from pathlib import Path
from tools.helper import read_specific_columns

'''
Takes the dataframe output by neighbor_accuracy.py and generates a graph (initially used to change alpha value without regenerating all the data). Additionally, the frequency of the number of neighbors and associated score is calculated and printed. 
'''

ranked = read_specific_columns("./output/data/go_neighbor_tests/neighbor_rank/all_neighbor_rank_under_100.csv", [1,2], "\t")

all_go_neighbor_rank = []
all_go_neighbor_num = []

# Sorts through data, remvoing values with no score
for i in ranked:
    if i[1] != "N/A":
        all_go_neighbor_num.append(int(i[0]))
        all_go_neighbor_rank.append(float(i[1]))

fig, ax = plt.subplots()
plt.scatter(all_go_neighbor_rank, all_go_neighbor_num, alpha = .05)
plt.xlabel("% Go Neighbors Accurately Predicted")
ax.set_xlim([-5, 105])
plt.ylabel("Number of Neighbors")
plt.savefig("./output/data/go_neighbor_tests/neighbor_rank/under_100_rank.png")
plt.show()

mean_rank = round(stat.mean(all_go_neighbor_rank),2)
mean_num = round(stat.mean(all_go_neighbor_num),2)
print("Mean Rank: " + str(mean_rank))
print("Mean Number of Neighbors: " + str(mean_num))

#Prints the frequency of some number of neighbors having a specific score, can uncomment below to remove any that have a score of zero
freq_dict = {}
for i in ranked:
    if i[1] != "N/A": # and float(i[1]) != 0.0:
        key = i[0] + "_" + i[1]
        if key in freq_dict.keys():
            freq_dict[key] += 1
        else:
            freq_dict[key] = 1

# Recursive function that saves the frequency output as a dictionary 
def order(freq_dict):
    top_freq = 0
    top = ""
    for i in freq_dict.keys():
        if freq_dict[i] > top_freq:
            top_freq = freq_dict[i]
            top = i 

    s = top.split("_")
    print(s[0] + " Neighbors with a percent accuracy of " + s[1] + " occurs " + str(top_freq) + " times")
    freq_dict.pop(top)
    if len(freq_dict) != 1:
        x = order(freq_dict)
        x.insert(0,[top, top_freq])
        return x
    return [[top, top_freq]]

lst = order(freq_dict)
print(lst)

#Dictionary is not saved anywhere but could be 
        
    
    