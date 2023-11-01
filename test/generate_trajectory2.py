import os
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

Sim = False  

fp = "data/RTK_read_data/July24/test.csv"

trajectory = np.loadtxt(fp, skiprows=2, delimiter=',')

fp_origin = "data/RTK_read_data/July24/origin.csv"

origin = np.loadtxt(fp_origin, skiprows=1, delimiter=',')

traj = trajectory[:, :2] - \
    np.array(origin)

X, Y = traj[:, 0], traj[:, 1]
X_filter = X[::10]
Y_filter = Y[::10]
tck, u = interpolate.splprep([X_filter, Y_filter], s=0.0)
x_i, y_i = interpolate.splev(np.linspace(0, 1, 100), tck)
Inter_traj = np.vstack((x_i, y_i)).T

if Sim:
    plt.plot(Inter_traj[:, 0], Inter_traj[:, 1])
    plt.xlabel("x axis /m")
    plt.ylabel("y axis /m")
    plt.title('processed trajectory')
    plt.show()

if not Sim:
    heading = []
    fields = ['x_position', 'y_position']
    with open('data/RTK_read_data/July24/keypoints.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(Inter_traj)
