import numpy as np
#############################################
# local path planning in frenet frame params
#############################################

# Simulation parameter

DT = 0.1  # time tick [s]
L_W = 1.25  # [m] Wheel base of vehicle
W_V = 1.0  # [m] width of vehicle
L_V = 2.1  # [m] total length of the vehicle
TARGET_SPEED = 3.6 / 3.6  # target speed [m/s]
ROBOT_RADIUS = 1.0  # robot radius [m]
MAX_ROAD_WIDTH = 6.0  # maximum road width [m]
MAX_SPEED = 7.2 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 5.0  # maximum acceleration [m/ss]
MAX_STEER = np.radians(45.0)  # [rad] max steering angle
MAX_SIMULATION_TIME = 200
# Local path planning parameter

D_ROAD_W = 1.0  # road width sampling length [m]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.9  # min prediction time [m]
D_T_S = 1.8 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed

# Cost function parameters

K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0
K_OBS = 5.0

# Stanley controller parameter

K_S = 0.1  # control gain
KP_S = 1.0  # speed proportional gain
