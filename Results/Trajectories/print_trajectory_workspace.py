import matplotlib.pyplot as plt
import json
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 7),
         'axes.labelsize': '25',
         'axes.titlesize':'25',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

with open("./Data/trajectory_EmbeddedControl.txt", 'r') as fp:
    load_data =  json.load(fp)
    x_learned, y_learned, z_learned = load_data["pos_effector"]["x"], load_data["pos_effector"]["y"], load_data["pos_effector"]["z"]
    x_goal, y_goal, z_goal = load_data["pos_goal"]["x"], load_data["pos_goal"]["y"], load_data["pos_goal"]["z"]



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs=x_goal, ys=y_goal, zs=z_goal, color = "orange", s=100)
ax.scatter(xs=x_learned, ys=y_learned, zs=z_learned, color = "red", marker='+', s=200)
ax.set_xlabel('X ')
ax.set_ylabel('Y ')
ax.set_zlabel('Z ')
plt.show()
