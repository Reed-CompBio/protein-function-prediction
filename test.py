import os
from pathlib import Path
a = {}
keys = ["A", "B", "C", "D"]
c = [0.6294894999999999, 
     0.7694035, 
     0.770448, 
     0.8242565, 
     0.6282075, 
     0.8592135000000001, 
     0.507865, 
     0.6913925000000001, 
     0.7600895]
for i in keys:
    a[i] = c

# print(a)

x = 5
dataset_directory_path = Path("./output/dataset")

def use_existing_samples(dataset_directory_path):
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
    r = range(n)
    for i in r:
        if i not in nums:
            n = nums[-1]
            pos = "rep_" + str(i) + "_positive_protein_go_term_pairs.csv"
            neg = "rep_" + str(i) + "_negative_protein_go_term_pairs.csv"
            pos_old_path = Path(dataset_directory_path, "rep_" + str(n) + "_positive_protein_go_term_pairs.csv")
            neg_old_path = Path(dataset_directory_path, "rep_" + str(n) + "_negative_protein_go_term_pairs.csv")
            pos_new_path = Path(dataset_directory_path, pos)
            neg_new_path = Path(dataset_directory_path, neg)
            os.rename(pos_old_path, pos_new_path)
            os.rename(neg_old_path, neg_new_path)
            nums.pop()
    return n

                    
print(use_existing_samples(dataset_directory_path))
