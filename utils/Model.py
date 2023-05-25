import pathlib
import sys

path = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(1, path)

import numpy as np

from utils.config import *


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


class State(object):
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, traj_d, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.steering_angle = 0
        self.v = v
        self.tau = 1
        self.a = 0
        self.k = 100

        self.s = 0
        self.d = 0
        self.dd = 0
        self.ddd = 0
        self.target_idx = 0
        self.traj_d = traj_d
        self.di_history = np.zeros(3)
        

    def update(self, acceleration, steering_command):
        """
        Update the state of the vehicle.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        self.di_history = np.roll(self.di_history, -1)
        self.di_history[-1] = steering_command

        delta = self.steering_angle + DT / self.tau * \
            (self.di_history[-1] - self.steering_angle)
        delta = np.clip(delta, -MAX_STEER, MAX_STEER)
        self.a = acceleration

        # delta = self.di_history[-1]
        # delta = np.clip(delta, -MAX_STEER, MAX_STEER)

        dx = self.v * np.cos(self.yaw)
        self.x += self.v * np.cos(self.yaw) * DT

        dy = self.v * np.sin(self.yaw)
        self.y += self.v * np.sin(self.yaw) * DT

        dyaw = self.v / L_W * np.tan(delta)
        self.yaw += self.v / L_W * np.tan(delta) * DT
        self.yaw = normalize_angle(self.yaw)

        dv = acceleration
        self.v += acceleration * DT

        ddx = dv * np.cos(self.yaw) - self.v * np.sin(self.yaw) * dyaw
        ddy = dv * np.sin(self.yaw) + self.v * np.cos(self.yaw) * dyaw
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2 + 1e-5)**(3 / 2))

        self.steering_angle = delta
        self.k = k

    def cart2frenet(self):

        cx = self.traj_d.cx
        cy = self.traj_d.cy
        s_list = self.traj_d.s

        fx = self.x
        fy = self.y
        # Search nearest point index
        try:
            dx = [fx - icx for icx in cx[self.target_idx:self.target_idx + 50]]
            dy = [fy - icy for icy in cy[self.target_idx:self.target_idx + 50]]

        except:
            dx = [fx - icx for icx in cx[self.target_idx:]]
            dy = [fy - icy for icy in cy[self.target_idx:]]

        d = np.hypot(dx, dy)
        target_idx = np.argmin(d) + self.target_idx

        if target_idx < 5:
            target_idx = 5

        v1 = np.array([fx - cx[target_idx], fy - cy[target_idx]])
        v2 = np.array([cx[target_idx] - cx[target_idx - 5],
                       cy[target_idx] - cy[target_idx - 5]])
        d_min = np.sign(np.cross(v2, v1)) * d[target_idx - self.target_idx]

        cyaw = self.traj_d.cyaw[target_idx]
        ck = self.traj_d.ck[target_idx]
        dck = self.traj_d.dck[target_idx]
        s = s_list[target_idx]

        dx = fx - cx[target_idx]
        dy = fy - cy[target_idx]
        delta_yaw = normalize_angle(self.yaw - cyaw)

        tandeltayaw = np.tan(delta_yaw)
        cosdeltayaw = np.cos(delta_yaw)

        oneMinusKRefd = 1 - ck * d_min
        d_prime = oneMinusKRefd * tandeltayaw
        sdot = self.v * cosdeltayaw / oneMinusKRefd
        dck = dck / sdot

        kRefdd = dck * d_min + ck * d_prime
        d_prime_prime = - kRefdd * tandeltayaw + oneMinusKRefd / cosdeltayaw / \
            cosdeltayaw * (self.k * oneMinusKRefd / cosdeltayaw - ck)

        deltayaw_prime = oneMinusKRefd / cosdeltayaw * self.k - ck
        sdotdot = (self.a * cosdeltayaw - sdot ** 2 *
                   (d_prime * deltayaw_prime - dck)) / oneMinusKRefd

        dd = sdot * d_prime
        ddd = d_prime_prime * sdot**2 + d_prime * sdotdot

        self.s = s
        self.d = d_min
        self.dd = d_prime
        self.ddd = d_prime_prime
        self.target_idx = target_idx
