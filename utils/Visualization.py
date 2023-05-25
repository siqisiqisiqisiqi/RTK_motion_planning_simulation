import pathlib
import sys

path = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(1, path)

import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import math

from utils.config import *
from utils.Path import FrenetPath

pi = math.pi


class visualization():
    def __init__(self, traj_d, ob):
        self.cx = traj_d.cx
        self.cy = traj_d.cy
        self.cyaw = traj_d.cyaw
        self.road_bound = []
        self.calc_road_bound()
        self.ob = ob

    def calc_rect_right_corner(self, theta, length, width, x, y):
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xy = np.array([[length / 2], [width / 2]])
        xy = np.matmul(rotation_matrix, xy)
        xy = np.array([[x], [y]]) - xy
        return xy

    def calc_road_bound(self):
        traj_len = len(self.cx)
        fp_upper_bound = FrenetPath()
        fp_lower_bound = FrenetPath()
        fp_upper_bound.x = [self.cx[i] + MAX_ROAD_WIDTH / 2 *
                            np.cos(self.cyaw[i] - pi / 2) for i in range(traj_len)]
        fp_upper_bound.y = [self.cy[i] + MAX_ROAD_WIDTH / 2 *
                            np.sin(self.cyaw[i] - pi / 2) for i in range(traj_len)]
        fp_lower_bound.x = [self.cx[i] - MAX_ROAD_WIDTH / 2 *
                            np.cos(self.cyaw[i] - pi / 2) for i in range(traj_len)]
        fp_lower_bound.y = [self.cy[i] - MAX_ROAD_WIDTH / 2 *
                            np.sin(self.cyaw[i] - pi / 2) for i in range(traj_len)]
        self.road_bound.append(fp_upper_bound)
        self.road_bound.append(fp_lower_bound)

    def show_animation(self, state, path, stanley_idx, traj_actual):
        ax = plt.gca()
        plt.cla()
        # plot the vehicle
        xy = self.calc_rect_right_corner(state.yaw, L_V, W_V, state.x, state.y)
        rect = Rectangle((xy[0, 0], xy[1, 0]), L_V, W_V, angle=state.yaw *
                         180 / pi, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        # plot the obstacle
        for i in range(self.ob.shape[0]):
            ob = self.ob[i]
            xy = self.calc_rect_right_corner(
                ob[4], ob[2], ob[3], ob[0], ob[1])
            rect = Rectangle((xy[0, 0], xy[1, 0]), ob[2], ob[3], angle=ob[4]
                             * 180 / pi, linewidth=2, edgecolor='c', facecolor='c')
            ax.add_patch(rect)

        # plot the local path
        plt.plot(path.x[1:], path.y[1:], "-om")

        # plot the target path
        plt.plot(path.x[stanley_idx], path.y[stanley_idx], "xk")

        # plot the road bound
        plt.plot(self.road_bound[0].x, self.road_bound[0].y,
                 "k", linewidth=4)
        plt.plot(self.road_bound[1].x, self.road_bound[1].y, "k", linewidth=4)

        # plot the reference trajectory
        plt.plot(self.cx, self.cy, "--r", label="course")

        # plot the covered trajectory
        plt.plot(traj_actual.x, traj_actual.y, "-b")

        plt.axis("equal")
        plt.grid(True)
        plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
        plt.pause(0.001)

    def show_figure(self, traj_actual, delta_d_list, delta_r_list):
        ax = plt.gca()
        plt.plot(self.cx, self.cy, ".r")
        plt.plot(traj_actual.x, traj_actual.y, "-b", label="trajectory")
        plt.plot(self.road_bound[0].x, self.road_bound[0].y,
                 "k", linewidth=4, label="traj_bound")
        plt.plot(self.road_bound[1].x, self.road_bound[1].y, "k", linewidth=4)
        plt.plot(self.ob[:, 0], self.ob[:, 1], "xk")

        for i in range(self.ob.shape[0]):
            ob = self.ob[i]
            xy = self.calc_rect_right_corner(
                ob[4], ob[2], ob[3], ob[0], ob[1])
            rect = Rectangle((xy[0, 0], xy[1, 0]), ob[2], ob[3], angle=ob[4]
                             * 180 / pi, linewidth=2, edgecolor='c', facecolor='c')
            ax.add_patch(rect)

        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(False)

        plt.subplots(1)
        plt.plot(traj_actual.t, [iv * 3.6 for iv in traj_actual.v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(traj_actual.t, delta_d_list, "-r")
        plt.plot(traj_actual.t, delta_r_list, "-b")
        plt.legend(["desired steering angle", "actual steering angle"])
        plt.xlabel("Time[s]")
        plt.ylabel("steering angle")
        plt.grid(True)
        plt.show()
