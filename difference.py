from tools.helper import read_specific_columns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path
import numpy as np
import pandas as pd
from decimal import Decimal

#Compares inferred ROC and PR values given two datasets, code requires both to have been run with the same algorithms and saved in an auc_values_<name> file

a = pd.read_csv("./output/data/auc_values_fly.csv")
a_name = "Fly"
b = pd.read_csv("./output/data/auc_values_bsub.csv")
b_name = "B. Subtilis"

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
    if val > .09:
        return "royalblue"
    elif val > .03:
        return "cornflowerblue"
    elif val > 0:
        return "lightsteelblue"
    elif val == 0:
        return "white"
    elif val < -0.09:
        return "red"
    elif val < -0.03:
        return "tomato"
    else:
        return "lightsalmon"

color = [] #darker red -> b is better, darker blue -> a is better 
for i in difference:
    cell_color = ["","","",""]
    cell_color[0] = cellColor(difference[i][0])
    cell_color[1] = cellColor(-difference[i][1])
    cell_color[2] = cellColor(difference[i][2])
    cell_color[3] = cellColor(-difference[i][3])
    color.append(cell_color)

title = "The Difference Between " + a_name + " (Blue) and " + b_name + " (Red) Datasets"
savename = "auc_value_difference_" + a_name + "_" + b_name
savename = savename.lower().replace(" ", "")

fig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
tab = ax.table(cellText=df.values, rowLabels = rows, colLabels=cols, cellColours = color, bbox = [0, .2, 1, .6])
ax.set_title(title, x = .3, y = .82)
tab.auto_set_font_size(False)
tab.set_fontsize(7)
fig.tight_layout()
plt.savefig("./output/images/%s.png" % savename)
plt.show()

df.to_csv(
    Path("./output/data/", "%s.csv" % savename),
    index=True,
    sep="\t",
)


