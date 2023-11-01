import os
import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy import interpolate

path = os.getcwd()
os.chdir(path)

flag = "test"  # flag = "save" to save the keypoints

start_index = 800
end_index = 3000

fp = "data/RTK_read_data/test2.csv"

trajectory = np.loadtxt(fp, skiprows=2, delimiter=',')

X, Y, Z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
traj = trajectory[start_index:end_index, :2] - \
    np.array([X[start_index], Y[start_index]])
start_point = []

# plot the original trajectory
# plt.plot(traj[:,0],traj[:,1])
# plt.xlabel("x axis /m")
# plt.ylabel("y axis /m")
# plt.title('original trajectory')
# plt.show()

# split the trajectory
for index, (x, y) in enumerate(traj):
    if np.linalg.norm([x, y]) < 1:
        start_point.append(index)
start_point = np.array(start_point)
clustering = DBSCAN(eps=1, min_samples=2).fit(start_point.reshape(-1, 1))

sp = []  # seperate start point list
lp = []  # same start point list
index = 0
for i, label in enumerate(clustering.labels_):
    if label != index:
        index = label
        sp.append(lp)
        lp = []
    lp.append(start_point[i])
sp.append(lp)
process_start_index = [int(sum(p) / len(p)) for p in sp]
process_start_index[0] = 0
print(process_start_index)

circles = []
circle_num = len(process_start_index) - 1
for i in range(circle_num):
    circle_trajectory = traj[process_start_index[i]:process_start_index[i + 1]]
    circles.append(circle_trajectory)

# plot the processed trajectory
# circle = circles[0]
# plt.plot(circle[:,0], circle[:,1])
# plt.xlabel("x axis /m")
# plt.ylabel("y axis /m")
# plt.title('2D plot')
# plt.show()

# interpolate the trajectory and fuse the trajectory
interpolate_circles = []
for circle in circles:
    x, y = circle[:, 0], circle[:, 1]
    x = x[0::10]
    y = y[0::10]

    tck, u = interpolate.splprep([x, y], s=0.0)
    x_i, y_i = interpolate.splev(np.linspace(0, 1, 100), tck)

    interpolate_circles.append(np.vstack((x_i, y_i)).T)
# print(x_i.shape)interpolate_circles
# print(interpolate_circles.shape)
circle = sum(interpolate_circles) / len(interpolate_circles)

circle = interpolate_circles[1]
# plt.plot(circle[:,0], circle[:,1])
# plt.xlabel("x axis /m")
# plt.ylabel("y axis /m")
# plt.title('processed trajectory')
# plt.show()

keypoints = circle[0::4, :]
if flag == "save":
    heading = []
    fields = ['x_position', 'y_position']
    with open('data/RTK_read_data/keypoints_straight.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(circle)

# origin = np.array([[X[start_index], Y[start_index]]])
# heading = []
# fields = ['x_position', 'y_position']
# with open('data/RTK_read_data/origin.csv', 'w') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)

#     write.writerow(fields)
#     write.writerows(origin)


# plt.scatter(keypoints[:,0], keypoints[:,1], marker='o', color="red")
# plt.scatter(-19.3, 16.1, marker='o', color="red")
# plt.scatter(-19.3, 16.1, marker='o', color="blue")
plt.plot(circle[:, 0], circle[:, 1])
plt.xlabel("x axis /m")
plt.ylabel("y axis /m")
plt.title('processed trajectory')
plt.show()
