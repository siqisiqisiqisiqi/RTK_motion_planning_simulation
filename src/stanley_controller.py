
import pathlib
import sys

path = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(1, path)

from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
import math

from utils.CubicSpline import cubic_spline_planner

pi = math.pi
# show_animation = True
show_animation = True
max_steer = np.radians(45.0)  # [rad] max steering angle

k = 0.1  # control gain
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time difference
L = 1.25  # [m] Wheel base of vehicle
W = 1.0  # [m] width of vehicle
Length = 2.1  # [m] total length of the wehicle


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

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.steering_angle = 0
        self.v = v
        self.tau = 2

    def update(self, acceleration, steering_command):
        """
        Update the state of the vehicle.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """

        # delta = steering_command[-1]
        delta = self.steering_angle + dt / self.tau * \
            (steering_command[-1] - self.steering_angle)
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt
        self.steering_angle = delta


class Stanley_Controller:
    """

    Path tracking with Stanley steering control and PID speed control.
    Input:
        fp: trajecory file path
    Output:
        delta: Steering angle in radian

    author: Siqi Zheng

    Ref:
        - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
        - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

    """

    def __init__(self, fp):
        trajectory = np.loadtxt(fp, skiprows=2, delimiter=',')
        ax, ay = trajectory[:, 0], trajectory[:, 1]

        cx, cy, cyaw, _, _, _, _ = cubic_spline_planner.calc_spline_course(
            ax, ay, ds=0.1)
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.trajectory_length = len(cx)

    def pid_control(self, target, current):
        """
        Proportional control for the speed.

        :param target: (float)
        :param current: (float)
        :return: (float)
        """
        return Kp * (target - current)

    def calc_target_index(self, state):
        """
        Compute index in the trajectory list of the target.

        :param state: (State object)
        :param cx: [float]
        :param cy: [float]
        :return: (int, float)
        """
        # Calc front axle position
        fx = state.x + L * np.cos(state.yaw)
        fy = state.y + L * np.sin(state.yaw)

        # Search nearest point index
        dx = [fx - icx for icx in self.cx]
        dy = [fy - icy for icy in self.cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                          -np.sin(state.yaw + np.pi / 2)]
        error_front_axle = np.dot(
            [dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle

    def stanley_control(self, state, last_target_idx, diff_angle):
        """
        Stanley steering control.

        :param state: (State object)
        :param cx: ([float])
        :param cy: ([float])
        :param cyaw: ([float])
        :param last_target_idx: (int)
        :return: (float, int)
        """
        current_target_idx, error_front_axle = self.calc_target_index(state)

        if last_target_idx >= current_target_idx:
            current_target_idx = last_target_idx

        # theta_e corrects the heading error
        theta_e = normalize_angle(self.cyaw[current_target_idx] - state.yaw)
        # theta_d corrects the cross track error
        theta_d = 0.5*np.arctan2(k * error_front_axle, np.maximum(state.v, 0.2))
        # Steering control
        # delta = theta_e + theta_d + 0 * diff_angle
        delta = theta_e + theta_d + 0.7 * diff_angle
        return delta, current_target_idx


def main():
    """Plot an example of Stanley steering control on a cubic spline."""
    #  target course
    fp = "data/RTK_read_data/keypoints.csv"

    trajectory = np.loadtxt(fp, skiprows=2, delimiter=',')
    ax, ay = trajectory[:, 0], trajectory[:, 1]

    cx, cy, _, _, _, _, _ = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)
    # print(cx)

    # target_speed = 3.6 / 3.6  # [m/s]

    target_speed = 7.2 / 3.6

    max_simulation_time = 200

    # Initial state
    state = State(x=0, y=0, yaw=0, v=0.0)

    last_idx = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]

    di_history = np.zeros(3)

    stanleycontroller = Stanley_Controller(fp)
    target_idx, _ = stanleycontroller.calc_target_index(state)

    steering_angle_command_history = []
    actual_steering_angle_history = []
    time_history = []
    diff_angle = 0

    while max_simulation_time >= time and last_idx > target_idx:
        # print(target_idx)

        ai = stanleycontroller.pid_control(target_speed, state.v)

        di, target_idx = stanleycontroller.stanley_control(
            state, target_idx, diff_angle)

        di_history = np.roll(di_history, -1)
        di_history[-1] = di

        state.update(ai, di_history)

        steering_angle_command_history.append(di)
        actual_steering_angle_history.append(state.steering_angle)
        time_history.append(time)

        diff_angle = di - state.steering_angle

        time += dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        theta = state.yaw
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xy = np.array([[Length / 2], [W / 2]])
        xy = np.matmul(rotation_matrix, xy)
        xy = np.array([[state.x], [state.y]]) - xy

        # if show_animation:  # pragma: no cover
        #     ax = plt.gca()
        #     plt.cla()
        #     rect = Rectangle((xy[0, 0], xy[1, 0]), Length, W, angle=state.yaw *
        #                      180 / pi, linewidth=2, edgecolor='b', facecolor='none')
        #     # for stopping simulation with the esc key.
        #     plt.gcf().canvas.mpl_connect('key_release_event',
        #                                  lambda event: [exit(0) if event.key == 'escape' else None])
        #     plt.plot(cx, cy, ".r", label="course")
        #     plt.plot(state.x, state.y, ".b")
        #     plt.plot(x, y, "-b", label="trajectory")
        #     plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
        #     ax.add_patch(rect)
        #     plt.axis("equal")
        #     plt.grid(True)
        #     plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
        #     plt.pause(0.001)

    # Test
    assert last_idx >= target_idx, "Cannot reach goal"

    if show_animation:  # pragma: no cover
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(t, [iv * 3.6 for iv in v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(time_history, steering_angle_command_history, "-r")
        plt.plot(time_history, actual_steering_angle_history, "-b")
        plt.legend(["desired steering angle", "actual steering angle"])
        plt.xlabel("Time[s]")
        plt.ylabel("steering angle")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()
