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



parser = argparse.ArgumentParser('Process args')
parser.add_argument('--name', '-n', help='Load a model: -n model_name')
parser.add_argument('--x', '-x', help='The value of x-axis')
parser.add_argument('--y', '-y', help='The value of y-axis')
args = parser.parse_args(None)

data = {"data_true": {}, "data_learned": {}, "data_computed": {}}

print("load: ./Data/trajectory_"+args.name+".txt")
with open("./Data/trajectory_"+args.name+"_learned.txt", 'r') as fp:
    load_data =  json.load(fp)
    data["data_learned"], data["data_true"] = load_data["pos_effector"], load_data["pos_goal"]
with open("./Data/trajectory_"+args.name+"_computed.txt", 'r') as fp:
    load_data =  json.load(fp)
    data["data_computed"] = load_data["pos_effector"]


data["data_true"]["t"] = list(range(len(data["data_true"]["x"])))
data["data_learned"]["t"] = list(range(len(data["data_learned"]["x"])))
data["data_computed"]["t"] = list(range(len(data["data_computed"]["x"])))

pdData = pd.DataFrame.from_dict(data)
sns.scatterplot(data=pdData["data_true"], x=args.x, y=args.y, color = "orange", s=100)
sns.scatterplot(data=pdData["data_learned"], x=args.x, y=args.y, color = "blue", marker='x', s=200)
sns.scatterplot(data=pdData["data_computed"], x=args.x, y=args.y, color = "red", marker='+', s=200)
# plt.legend(("Goals", "Trajectory with learned values", "Trajectory with computed values"))
plt.show()
