import pathlib
import sys
import copy

path = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(1, path)

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import math

from utils.QuinticPolynomialsPlanner.quintic_polynomials_planner import \
    QuinticPolynomial
from utils.CubicSpline import cubic_spline_planner
from utils.Path import FrenetPath, DesiredCartesianTrajectory
from utils.QuarticPolynomial import QuarticPolynomial
from obb_collision_detection import calculate_vertice, collide
from config import *

pi = math.pi
show_animation = True


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


def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH / 2, MAX_ROAD_WIDTH / 2 + D_ROAD_W, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

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
                    s0, c_speed, c_accel, tv, 0.0, Ti / tv)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t / tv]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t / tv]
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


def obb_collision_detection(fp, ob):
    d_list = []
    for i in range(len(ob[:, 0])):
        d = [((fp.x[1] - ob[i, 0]) ** 2 + (fp.y[1] - ob[i, 1]) ** 2)]
        d_list.append(d)
    i = np.argmin(d_list)
    ob1 = [ob[i, 0], ob[i, 1], ob[i, 4], ob[i, 3], ob[i, 2]]
    v1 = calculate_vertice(ob1)
    d_min = 255
    i = 0
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


def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]
        collision = any([di <= (W_V / 2 + ob[i, 3]) ** 2 for di in d])
        if collision:
            return False

    return True


def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        collision, d = obb_collision_detection(fplist[i], ob)
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            print("exceed max speed")
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            print("exceed max acceleration")
            continue
        # elif not check_collision(fplist[i], ob):
        #     continue
        elif not collision:
            continue
        fplist[i].cf = fplist[i].cf + K_OBS * 1 / (d - 1)
        ok_ind.append(fplist[i])
    return ok_ind


def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob, road_bound):
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

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


def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
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
        self.tau = 1
        self.a = 0
        self.k = 100

    def update(self, acceleration, steering_command):
        """
        Update the state of the vehicle.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """

        delta = self.steering_angle + DT / self.tau * \
            (steering_command[-1] - self.steering_angle)
        delta = np.clip(delta, -MAX_STEER, MAX_STEER)
        self.a = acceleration
        # delta = steering_command[-1]
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


def coordinate_transform(traj_d, state, last_target_idx):

    cx = traj_d.cx
    cy = traj_d.cy
    s_list = traj_d.s

    fx = state.x
    fy = state.y
    # Search nearest point index
    try:
        dx = [fx - icx for icx in cx[last_target_idx:last_target_idx + 50]]
        dy = [fy - icy for icy in cy[last_target_idx:last_target_idx + 50]]

    except:
        dx = [fx - icx for icx in cx[last_target_idx:]]
        dy = [fy - icy for icy in cy[last_target_idx:]]

    d = np.hypot(dx, dy)
    target_idx = np.argmin(d) + last_target_idx

    if target_idx < 5:
        target_idx = 5

    v1 = np.array([fx - cx[target_idx], fy - cy[target_idx]])
    v2 = np.array([cx[target_idx] - cx[target_idx - 5],
                  cy[target_idx] - cy[target_idx - 5]])
    d_min = np.sign(np.cross(v2, v1)) * d[target_idx - last_target_idx]

    cyaw = traj_d.cyaw[target_idx]
    ck = traj_d.ck[target_idx]
    dck = traj_d.dck[target_idx]
    s = s_list[target_idx]

    dx = fx - cx[target_idx]
    dy = fy - cy[target_idx]
    delta_yaw = normalize_angle(state.yaw - cyaw)

    tandeltayaw = np.tan(delta_yaw)
    cosdeltayaw = np.cos(delta_yaw)

    oneMinusKRefd = 1 - ck * d_min
    d_prime = oneMinusKRefd * tandeltayaw
    sdot = state.v * cosdeltayaw / oneMinusKRefd
    dck = dck / sdot

    kRefdd = dck * d_min + ck * d_prime
    d_prime_prime = - kRefdd * tandeltayaw + oneMinusKRefd / cosdeltayaw / \
        cosdeltayaw * (state.k * oneMinusKRefd / cosdeltayaw - ck)

    deltayaw_prime = oneMinusKRefd / cosdeltayaw * state.k - ck
    sdotdot = (state.a * cosdeltayaw - sdot ** 2 *
               (d_prime * deltayaw_prime - dck)) / oneMinusKRefd

    dd = sdot * d_prime
    ddd = d_prime_prime * sdot**2 + d_prime * sdotdot
    return s, d_min, d_prime, d_prime_prime, target_idx


def calc_road_bound(cx, cy, cyaw, ck):
    road_bound = []
    traj_len = len(cx)
    fp_upper_bound = FrenetPath()
    fp_lower_bound = FrenetPath()
    fp_upper_bound.x = [cx[i] + MAX_ROAD_WIDTH / 2 *
                        np.cos(cyaw[i] - pi / 2) for i in range(traj_len)]
    fp_upper_bound.y = [cy[i] + MAX_ROAD_WIDTH / 2 *
                        np.sin(cyaw[i] - pi / 2) for i in range(traj_len)]
    fp_lower_bound.x = [cx[i] - MAX_ROAD_WIDTH / 2 *
                        np.cos(cyaw[i] - pi / 2) for i in range(traj_len)]
    fp_lower_bound.y = [cy[i] - MAX_ROAD_WIDTH / 2 *
                        np.sin(cyaw[i] - pi / 2) for i in range(traj_len)]
    road_bound.append(fp_upper_bound)
    road_bound.append(fp_lower_bound)
    return road_bound


def main():
    """Plot an example of Stanley steering control on a cubic spline."""
    #  target course
    fp = "data/RTK_read_data/keypoints.csv"

    trajectory = np.loadtxt(fp, skiprows=2, delimiter=',')
    ax, ay = trajectory[:, 0], trajectory[:, 1]

    ob = np.array([[9.5, 15, 1.5, 1.0, 120 * pi / 180], [-7, 28.3, 1.5,
                  1.5, 200 * pi / 180], [-17.2, 15, 4.5, 1.5, -75 * pi / 180]])

    cx, cy, cyaw, ck, dck, s_list, csp = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)

    traj_d = DesiredCartesianTrajectory(cx, cy, cyaw, ck, dck, s_list)

    road_bound = calc_road_bound(cx, cy, cyaw, ck)

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
    target_idx = 0

    steering_angle_command_history = []
    actual_steering_angle_history = []
    time_history = []
    diff_angle = 0

    c_speed = 0.0  # current speed [m/s]
    c_accel = 0.0  # current acceleration [m/ss]
    c_d = 0.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position

    d_list = []
    dd_list = []
    ddd_list = []
    while max_simulation_time >= time:

        path, fplist = frenet_optimal_planning(
            csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob, road_bound)

        ai = stanleycontroller.pid_control(path.s_d[-1], state.v)

        stanleycontroller.update_traj(path)

        di, stanley_idx = stanleycontroller.stanley_control(
            state, diff_angle)

        di_history = np.roll(di_history, -1)
        di_history[-1] = di

        state.update(ai, di_history)
        last_target_idx = target_idx
        s, d, dd, ddd, target_idx = coordinate_transform(
            traj_d, state, last_target_idx)

        d_list.append(d)
        dd_list.append(dd)
        ddd_list.append(ddd)

        s0 = s
        c_d = d
        # c_d_d = path.d_d[1]
        # c_d_dd = path.d_dd[1]
        c_d_d = dd
        c_d_dd = ddd

        c_speed = state.v
        c_accel = ai

        steering_angle_command_history.append(di)
        actual_steering_angle_history.append(state.steering_angle)
        time_history.append(time)

        diff_angle = di - state.steering_angle

        time += DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        theta = state.yaw
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xy = np.array([[L_V / 2], [W_V / 2]])
        xy = np.matmul(rotation_matrix, xy)
        xy = np.array([[state.x], [state.y]]) - xy

        if np.hypot(path.x[1] - cx[-1], path.y[1] - cy[-1]) <= 1.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            ax = plt.gca()
            plt.cla()
            rect = Rectangle((xy[0, 0], xy[1, 0]), L_V, W_V, angle=state.yaw *
                             180 / pi, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            for i in range(ob.shape[0]):
                theta = ob[i, 4]
                rotation_matrix = np.array(
                    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                xy = np.array([[ob[i, 2] / 2], [ob[i, 3] / 2]])
                xy = np.matmul(rotation_matrix, xy)
                xy = np.array([[ob[i, 0]], [ob[i, 1]]]) - xy
                rect = Rectangle((xy[0, 0], xy[1, 0]), ob[i, 2], ob[i, 3], angle=ob[i, 4]
                                 * 180 / pi, linewidth=2, edgecolor='c', facecolor='c')
                ax.add_patch(rect)

            plt.plot(path.x[1:], path.y[1:], "-om")
            plt.plot(path.x[stanley_idx], path.y[stanley_idx], "xk")
            # for fp in fplist:
            #     plt.plot(fp.x[1:], fp.y[1:])
            plt.plot(road_bound[0].x, road_bound[0].y,
                     "k", linewidth=4)
            plt.plot(road_bound[1].x, road_bound[1].y, "k", linewidth=4)
            plt.plot(cx, cy, "--r", label="course")
            plt.plot(state.x, state.y, ".b")
            plt.plot(x, y, "-b")
            plt.plot(path.x[target_idx - last_target_idx], path.y[target_idx - last_target_idx],
                     "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert last_idx >= target_idx, "Cannot reach goal"

    if show_animation:  # pragma: no cover
        ax = plt.gca()
        plt.plot(cx, cy, ".r")
        plt.plot(x, y, "-b", label="trajectory")
        plt.plot(road_bound[0].x, road_bound[0].y,
                 "k", linewidth=4, label="traj_bound")
        plt.plot(road_bound[1].x, road_bound[1].y, "k", linewidth=4)
        plt.plot(ob[:, 0], ob[:, 1], "xk")

        for i in range(ob.shape[0]):

            theta = ob[i, 4]
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            xy = np.array([[ob[i, 2] / 2], [ob[i, 3] / 2]])
            xy = np.matmul(rotation_matrix, xy)
            xy = np.array([[ob[i, 0]], [ob[i, 1]]]) - xy
            rect = Rectangle((xy[0, 0], xy[1, 0]), ob[i, 2], ob[i, 3], angle=ob[i, 4]
                             * 180 / pi, linewidth=2, edgecolor='c', facecolor='c')
            ax.add_patch(rect)

        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(False)

        # plt.subplots(1)
        # plt.plot(t, [iv * 3.6 for iv in v], "-r")
        # plt.xlabel("Time[s]")
        # plt.ylabel("Speed[km/h]")
        # plt.grid(True)

        # plt.subplots(1)
        # plt.plot(time_history, steering_angle_command_history, "-r")
        # plt.plot(time_history, actual_steering_angle_history, "-b")
        # plt.legend(["desired steering angle", "actual steering angle"])
        # plt.xlabel("Time[s]")
        # plt.ylabel("steering angle")
        # plt.grid(True)
        plt.show()

    # diff_dd = np.diff(dd_list)/0.1
    # plt.plot(diff_dd, label = "dd_")
    # plt.plot(d_list, label = "d")
    # plt.plot(dd_list, label = "dd")
    # plt.plot(ddd_list, label = "ddd")
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
