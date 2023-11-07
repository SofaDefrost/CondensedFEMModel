import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import json
import time
import pandas as pd

import argparse

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 7),
         'axes.labelsize': '25',
         'axes.titlesize':'25',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


data = []
idx = [0, 1, 2, 3]
color = ["blue", "red", "green", "cyan"]

for i in idx:
    name_learned = "./Data/trajectory_FingerElasticityParams_learned_" + str(i)+".txt"
    name_computed = "./Data/trajectory_FingerElasticityParams_computed_" + str(i)+".txt"
    d = {"data_true": {}, "data_learned": {}, "data_computed": {}}


    with open(name_learned, 'r') as fp:
        load_data =  json.load(fp)
        d["data_learned"], d["data_true"] = load_data["pos_effector"], load_data["pos_goal"]
    with open(name_computed, 'r') as fp:
        load_data =  json.load(fp)
        d["data_computed"] = load_data["pos_effector"]

    d["data_true"]["t"] = list(range(len(d["data_true"]["x"])))
    d["data_learned"]["t"] = list(range(len(d["data_learned"]["x"])))
    d["data_computed"]["t"] = list(range(len(d["data_computed"]["x"])))

    data.append(d)



    pdData = pd.DataFrame.from_dict(d)
    if i == 1:
        sns.scatterplot(data=pdData["data_true"], x="y", y="z", color = "orange", s=100)
    sns.scatterplot(data=pdData["data_learned"], x="y", y="z", color = color[i], marker='x', s=200)
    sns.scatterplot(data=pdData["data_computed"], x="y", y="z", color = color[i], marker='+', s=200)


# plt.legend(("Goals", "Trajectory with learned values", "Trajectory with computed values"))
plt.show()
