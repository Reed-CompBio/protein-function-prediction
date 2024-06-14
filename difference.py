from tools.helper import read_specific_columns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path
import numpy as np
import pandas as pd
from decimal import Decimal

#Compares inferred ROC and PR values to values from a non-inferred dataset, code requires both to have been run with the same algorithms and for existing auc_values.csv file and auc_values_no_inferred_edges.csv

a = pd.read_csv("./output/data/auc_values_fly.csv")
b = pd.read_csv("./output/data/auc_values_no_inferred_edges.csv")

k = list(a)[0]
keys = k.split('\t')
keys.remove('')

k2 = list(b)[0]
keys2 = k2.split('\t')
keys2.remove('')

In = {}
NIn = {}

for i in a[k].keys():
    temp = a[k][i]
    temp = temp.split('\t')
    In[temp[0]] = [Decimal(temp[1]), Decimal(temp[2]), Decimal(temp[3]), Decimal(temp[4])]

    temp = b[k][i]
    temp = temp.split('\t')
    NIn[temp[0]] = [Decimal(temp[1]), Decimal(temp[2]), Decimal(temp[3]), Decimal(temp[4])]

difference = {}

for i in In:
   difference[i] = [In[i][0] - NIn[i][0], In[i][1] - NIn[i][1], In[i][2] - NIn[i][2], In[i][3] - NIn[i][3]]

rows = list(difference.keys())
cols = ["ROC mean", "ROC sd", "Precision/Recall \nmean", "Precision/Recall \nsd"]

df = pd.DataFrame.from_dict(
    difference,
    orient="index",
    columns= cols,
)

def cellColor(val):
    if val > .9:
        return "darkolivegreen"
    elif val > .3:
        return "olive"
    elif val > 0:
        return "yellowgreen"
    elif val == 0:
        return "white"
    elif val < -0.9:
        return "darkred"
    elif val < -0.3:
        return "firebrick"
    else:
        return "indianred"

color = [] #More red -> b is better, more green -> a is better 
for i in difference:
    cell_color = ["","","",""]
    cell_color[0] = cellColor(difference[i][0])
    cell_color[1] = cellColor(difference[i][1])
    cell_color[2] = cellColor(difference[i][2])
    cell_color[3] = cellColor(difference[i][3])
    color.append(cell_color)

fig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
tab = ax.table(cellText=df.values, rowLabels = rows, colLabels=cols, cellColours = color, bbox = [0, .2, 1, .6])
tab.auto_set_font_size(False)
tab.set_fontsize(7)
fig.tight_layout()
plt.show()

df.to_csv(
    Path("./output/data/", "difference_between_inferred_and_not.csv"),
    index=True,
    sep="\t",
)


