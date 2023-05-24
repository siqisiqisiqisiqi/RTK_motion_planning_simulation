import numpy as np

# MAX_ROAD_WIDTH = 5.0
# D_ROAD_W = 0.5
# for di in np.arange(-MAX_ROAD_WIDTH/2, MAX_ROAD_WIDTH/2+D_ROAD_W, D_ROAD_W):
#     print(di)
TARGET_SPEED = 7.2 / 3.6  # target speed [m/s]
D_T_S = 5 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                    TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
    print(tv)
