import pathlib
import sys
import copy

path = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(1, path)

import numpy as np
import math

from utils.QuinticPolynomialsPlanner.quintic_polynomials_planner import \
    QuinticPolynomial
from utils.CubicSpline import cubic_spline_planner
from utils.Path import FrenetPath, DesiredCartesianTrajectory
from utils.QuarticPolynomial import QuarticPolynomial
from obb_collision_detection import calculate_vertice, collide
from utils.Visualization import visualization
from utils.config import *
from utils.Model import State

pi = math.pi
show_animation = True
show_figure = True


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


class FrenetPathPlanning():
    """Frenet path planning algorithm
    """
    def __init__(self, ob, traj_d) -> None:
        self.ob = ob
        self.traj_d = traj_d

    def frenet_optimal_planning(self, state):
        fplist = self.calc_frenet_paths(state)
        fplist = self.calc_global_paths(fplist)
        fplist = self.check_paths(fplist)

        # find minimum cost path
        min_cost = float("inf")
        best_path = None
        for fp in fplist:
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path = fp

        if best_path == None:
            print("can't find any path!!!!!")
        return best_path, fplist

    def check_paths(self, fplist):
        ok_ind = []
        for i, _ in enumerate(fplist):
            collision, d = self.obb_collision_detection(fplist[i])
            if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
                print("exceed max speed")
                continue
            elif any([abs(a) > MAX_ACCEL for a in
                      fplist[i].s_dd]):  # Max accel check
                print("exceed max acceleration")
                continue
            elif not collision:
                continue
            fplist[i].cf = fplist[i].cf + K_OBS * 1 / (d - 1)
            ok_ind.append(fplist[i])
        return ok_ind

    def obb_collision_detection(self, fp):
        d_list = []
        d_min = 255
        i = 0
        for i in range(len(self.ob[:, 0])):
            d = [((fp.x[1] - self.ob[i, 0]) ** 2 + (fp.y[1] - self.ob[i, 1]) ** 2)]
            d_list.append(d)
        i = np.argmin(d_list)
        ob1 = [self.ob[i, 0], self.ob[i, 1],
               self.ob[i, 4], self.ob[i, 3], self.ob[i, 2]]
        v1 = calculate_vertice(ob1)
        for (ix, iy, yaw) in zip(fp.x, fp.y, fp.yaw):
            vehicle = [ix, iy, yaw, W_V, L_V]
            v2 = calculate_vertice(vehicle)
            detection, d = collide(v1, v2)
            if detection == True:
                return False, None
            if d < d_min and i < 10:
                d_min = d
            i = i + 1
        return True, d

    def calc_frenet_paths(self, state):
        frenet_paths = []

        # generate path to each offset goal
        for di in np.arange(-MAX_ROAD_WIDTH / 2, MAX_ROAD_WIDTH / 2 + D_ROAD_W, D_ROAD_W):

            # Lateral motion planning
            for Ti in np.arange(MIN_T, MAX_T, DT):
                fp = FrenetPath()

                # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
                lat_qp = QuinticPolynomial(
                    state.d, state.dd, state.ddd, di, 0.0, 0.0, Ti)

                fp.t = [t for t in np.arange(0.0, Ti, DT)]
                fp.d = [lat_qp.calc_point(t) for t in fp.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # Longitudinal motion planning (Velocity keeping)
                for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                    TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                    tfp = copy.deepcopy(fp)
                    lon_qp = QuarticPolynomial(
                        state.s, state.v, state.a, tv, 0.0, Ti / tv)

                    tfp.s = [lon_qp.calc_point(t) for t in fp.t / tv]
                    tfp.s_d = [lon_qp.calc_first_derivative(
                        t) for t in fp.t / tv]
                    tfp.s_dd = [lon_qp.calc_second_derivative(
                        t) for t in fp.t / tv]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(
                        t) for t in fp.t / tv]

                    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                    # square of diff from target speed
                    ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                    tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                    tfp.cv = K_J * Js + K_T * Ti / tv + K_D * ds * 2
                    tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                    frenet_paths.append(tfp)

        return frenet_paths

    def calc_global_paths(self, fplist):
        for fp in fplist:

            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = self.traj_d.csp.calc_position(fp.s[i])
                if ix is None:
                    break
                i_yaw = self.traj_d.csp.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)

            # calc yaw and ds
            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(math.hypot(dx, dy))

            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])

            # calc curvature
            for i in range(len(fp.yaw) - 1):
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / (fp.ds[i]) + 1e-5)

        return fplist


class Stanley_Controller:
    """

    Path tracking with Stanley steering control and PID speed control.
    Input:
        fp: trajecory file path
    Output:
        delta: Steering angle in radian

    """

    def __init__(self, fp):
        trajectory = np.loadtxt(fp, skiprows=2, delimiter=',')
        ax, ay = trajectory[:, 0], trajectory[:, 1]

        cx, cy, cyaw, ck, dck, s, scp = cubic_spline_planner.calc_spline_course(
            ax, ay, ds=0.1)

        self.cx_origin = cx
        self.cy_origin = cy

        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.trajectory_length = len(cx)
        self.diff_angle = 0

    def update_traj(self, path):
        self.cx = path.x
        self.cy = path.y
        self.cyaw = path.yaw

    def pid_control(self, target, current):
        """
        Proportional control for the speed.

        :param target: (float)
        :param current: (float)
        :return: (float)
        """
        return KP_S * (target - current)

    def calc_target_index(self, state):
        """
        Compute index in the trajectory list of the target.

        :param state: (State object)
        :param cx: [float]
        :param cy: [float]
        :return: (int, float)
        """
        # Calc front axle position
        fx = state.x + L_W * np.cos(state.yaw)
        fy = state.y + L_W * np.sin(state.yaw)

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

    def stanley_control(self, state, diff_angle):
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

        # theta_e corrects the heading error
        theta_e = normalize_angle(self.cyaw[current_target_idx] - state.yaw)
        # theta_d corrects the cross track error
        theta_d = 0.5 * np.arctan2(K_S * error_front_axle,
                                   np.maximum(state.v, 0.2))

        delta = theta_e + theta_d + 0.7 * diff_angle
        return delta, current_target_idx


def main():
    #  target course
    fp = "data/RTK_read_data/keypoints.csv"
    trajectory = np.loadtxt(fp, skiprows=2, delimiter=',')
    ax, ay = trajectory[:, 0], trajectory[:, 1]
    ob = np.array([[9.5, 15, 1.5, 1.0, 120 * pi / 180], [-7, 28.3, 1.5,
                  1.5, 200 * pi / 180], [-17.2, 15, 4.5, 1.5, -75 * pi / 180]])
    cx, cy, cyaw, ck, dck, s_list, csp = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)
    traj_d = DesiredCartesianTrajectory(cx, cy, cyaw, ck, dck, s_list, csp)

    # Initial state
    time = 0.0
    delta_d_list = []
    delta_r_list = []
    traj_actual = FrenetPath()
    visual = visualization(traj_d, ob)
    stanleycontroller = Stanley_Controller(fp)
    state = State(traj_d, x=0, y=0, yaw=0, v=0.0)
    frenet_optimal_planner = FrenetPathPlanning(ob, traj_d)

    while MAX_SIMULATION_TIME >= time:

        path, _ = frenet_optimal_planner.frenet_optimal_planning(state)
        stanleycontroller.update_traj(path)
        ai = stanleycontroller.pid_control(path.s_d[-1], state.v)
        di, stanley_idx = stanleycontroller.stanley_control(state, state.diff_angle)

        state.update(ai, di)
        state.cart2frenet()

        time += DT
        delta_d_list.append(di)
        delta_r_list.append(state.steering_angle)
        traj_actual.x.append(state.x)
        traj_actual.y.append(state.y)
        traj_actual.yaw.append(state.yaw)
        traj_actual.v.append(state.v)
        traj_actual.t.append(time)

        if np.hypot(path.x[1] - cx[-1], path.y[1] - cy[-1]) <= 1.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            visual.show_animation(state, path, stanley_idx, traj_actual)

    if show_figure:  # pragma: no cover
        visual.show_figure(traj_actual, delta_d_list, delta_r_list)


if __name__ == '__main__':
    main()
